from tkinter import Tk, BOTH, Button, RIGHT, Scrollbar
from tkinter.ttk import Frame, Treeview

from tha4.pytasuku.workspace import Workspace, PlaceholderTask


class TaskSelectorUi(Frame):
    def __init__(self, root, workspace: Workspace):
        super().__init__()
        self.root = root
        self.workspace = workspace
        self.master.title("Tasks")
        self.master.geometry("256x512")

        treeview_frame = Frame(self)
        treeview_frame.pack(fill=BOTH, expand=True)

        self.treeview = Treeview(treeview_frame)
        self.treeview["columns"] = ("task_name")
        self.treeview.column("#0", width=256, minwidth=256)
        self.treeview.heading("#0", text="Tree")
        self.treeview.heading("task_name", text="Task Name")

        treeview_vertical_scroll = Scrollbar(treeview_frame,
                                             orient='vertical',
                                             command=self.treeview.yview)
        self.treeview.configure(yscrollcommand=treeview_vertical_scroll.set)
        treeview_vertical_scroll.pack(side=RIGHT, fill='y')
        self.treeview.pack(fill=BOTH, expand=True)

        treeview_horizontal_scroll = Scrollbar(treeview_frame,
                                               orient='horizontal',
                                               command=self.treeview.xview)
        self.treeview.configure(xscrollcommand=treeview_horizontal_scroll.set)
        treeview_horizontal_scroll.pack(fill='x')

        self.add_tree_nodes()

        self.execute_button = Button(self, text="Execute!", command=self.run_selected_task)
        self.execute_button.pack(side=RIGHT, padx=5, pady=5)

        self.pack(fill=BOTH, expand=True)

        self.selected_task_name = None

    def add_tree_nodes(self):
        nodes = {}

        for task in self.workspace._tasks.values():
            if isinstance(task, PlaceholderTask):
                continue
            comps = task.name.split('/')
            for i in range(1, len(comps)):
                assert len(comps) > 0
            prefix = ""
            index = 0
            for comp in comps:
                index = index + 1
                parent = prefix
                if prefix == "" and comp == "":
                    prefix = "/"
                elif prefix == "":
                    prefix = prefix + comp
                elif prefix == "/":
                    prefix = prefix + comp
                else:
                    prefix = prefix + "/" + comp
                if prefix in nodes:
                    continue
                if index == len(comps):
                    data = prefix
                else:
                    data = ""
                if prefix == "/":
                    comp = "/"
                nodes[prefix] = {
                    "name": str(prefix),
                    "display_name": comp,
                    "parent": parent,
                    "data": data
                }

        sorted_node_names = sorted(nodes.keys())
        node_index = {}
        for name in sorted_node_names:
            node = nodes[name]
            if node["parent"] == "":
                id = self.treeview.insert("", "end", text=node["display_name"], values=node["data"], )
            else:
                parent = node_index[node["parent"]]
                id = self.treeview.insert(parent, "end", text=node["display_name"], values=node["data"], )
            node_index[node["name"]] = id

    def run_selected_task(self):
        selection = self.treeview.selection()
        item = self.treeview.item(selection)
        if item['values'] == "":
            return
        task_name = item["values"][0]
        self.selected_task_name = task_name
        self.root.destroy()


def run_task_selector_ui(workspace: Workspace):
    root = Tk()
    task_selector_ui = TaskSelectorUi(root, workspace=workspace)
    root.mainloop()

    task_name = task_selector_ui.selected_task_name
    if task_name is not None:
        print("Running", task_name, "...")
        with workspace.session():
            workspace.run(task_name)
