import copy
import logging
import os
from typing import Dict, Optional

import torch
from torch.nn import Module
from torch.optim import Optimizer

from tha4.shion.core.load_save import torch_save, torch_load
from tha4.shion.core.module_accumulator import ModuleAccumulator
from tha4.shion.core.module_factory import ModuleFactory
from tha4.shion.core.optimizer_factory import OptimizerFactory
from tha4.shion.core.training.util import optimizer_to_device


class TrainingState:
    def __init__(self,
                 examples_seen_so_far: int,
                 modules: Dict[str, Module],
                 accumulated_modules: Dict[str, Module],
                 optimizers: Dict[str, Optimizer]):
        self.accumulated_modules = accumulated_modules
        self.optimizers = optimizers
        self.modules = modules
        self.examples_seen_so_far = examples_seen_so_far

    @staticmethod
    def get_examples_seen_so_far_file_name(prefix) -> str:
        return prefix + "/examples_seen_so_far.txt"

    @staticmethod
    def get_module_file_name(prefix, module_name) -> str:
        return "%s/module_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_accumulated_module_file_name(prefix, module_name) -> str:
        return "%s/accumulated_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_optimizer_file_name(prefix, module_name) -> str:
        return "%s/optimizer_%s.pt" % (prefix, module_name)

    @staticmethod
    def get_rng_state_file_name(prefix):
        return "%s/rng_state.pt" % prefix

    def save(self, prefix):
        logging.info("Saving training state to %s" % prefix)
        os.makedirs(prefix, exist_ok=True)
        with open(TrainingState.get_examples_seen_so_far_file_name(prefix), "wt") as fout:
            fout.write("%d\n" % self.examples_seen_so_far)
            logging.info("Saved %s" % TrainingState.get_examples_seen_so_far_file_name(prefix))
        for module_name in self.modules:
            file_name = TrainingState.get_module_file_name(prefix, module_name)
            torch_save(self.modules[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        for module_name in self.accumulated_modules:
            file_name = TrainingState.get_accumulated_module_file_name(prefix, module_name)
            torch_save(self.accumulated_modules[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        for module_name in self.optimizers:
            file_name = TrainingState.get_optimizer_file_name(prefix, module_name)
            torch_save(self.optimizers[module_name].state_dict(), file_name)
            logging.info("Saved %s" % file_name)
        torch_save(torch.get_rng_state(), TrainingState.get_rng_state_file_name(prefix))
        logging.info("Saved %s" % TrainingState.get_rng_state_file_name(prefix))
        logging.info("Done saving training state to %s" % prefix)

    @staticmethod
    def get_examples_seen_so_far(prefix: str) -> int:
        with open(TrainingState.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            return int(lines[0])

    @staticmethod
    def load(prefix: str,
             module_factories: Dict[str, ModuleFactory],
             accumulators: Dict[str, ModuleAccumulator],
             optimizer_factories: Dict[str, OptimizerFactory],
             device: torch.device) -> 'TrainingState':
        logging.info("Loading training state from %s" % prefix)

        with open(TrainingState.get_examples_seen_so_far_file_name(prefix)) as fin:
            lines = fin.readlines()
            examples_seen_so_far = int(lines[0])
            logging.info("Loaded %s" % TrainingState.get_examples_seen_so_far_file_name(prefix))

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            file_name = TrainingState.get_module_file_name(prefix, module_name)
            modules[module_name].load_state_dict(torch_load(file_name))
            modules[module_name].to(device)
            logging.info("Loaded %s" % file_name)

        accumulated_modules = {}
        for module_name in accumulators:
            module_factory = module_factories[module_name]
            module = module_factory.create()
            file_name = TrainingState.get_accumulated_module_file_name(prefix, module_name)
            module.load_state_dict(torch_load(file_name))
            module.to(device)
            accumulated_modules[module_name] = module
            logging.info("Loaded %s" % file_name)

        optimizers = {}
        for module_name in optimizer_factories:
            optimizer = optimizer_factories[module_name].create(modules[module_name].parameters())
            file_name = TrainingState.get_optimizer_file_name(prefix, module_name)
            optimizer.load_state_dict(torch_load(file_name))
            optimizer_to_device(optimizer, device)
            optimizers[module_name] = optimizer
            logging.info("Loaded %s" % file_name)

        torch.set_rng_state(torch_load(TrainingState.get_rng_state_file_name(prefix)))
        logging.info("Loaded %s" % TrainingState.get_rng_state_file_name(prefix))

        logging.info("Done loading training state from %s" % prefix)

        return TrainingState(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def new(module_factories: Dict[str, ModuleFactory],
            accumulators: Dict[str, ModuleAccumulator],
            optimizer_factories: Dict[str, OptimizerFactory],
            random_seed: int,
            device: torch.device,
            pretrained_module_file_names: Optional[Dict[str, str]] = None) -> 'TrainingState':
        examples_seen_so_far = 0

        modules = {
            module_name: factory.create()
            for (module_name, factory) in module_factories.items()
        }
        for module_name in modules:
            modules[module_name].to(device)
        if pretrained_module_file_names is not None:
            for module_name in modules:
                if module_name in pretrained_module_file_names:
                    file_name = pretrained_module_file_names[module_name]
                    modules[module_name].load_state_dict(torch_load(file_name))
                    logging.info("Loaded initial state from %s ..." % file_name)

        accumulated_modules = {}
        for module_name in accumulators:
            accumulated_modules[module_name] = copy.deepcopy(modules[module_name])

        optimizers = {}
        for module_name in optimizer_factories:
            module = modules[module_name]
            optimizer = optimizer_factories[module_name].create(module.parameters())
            optimizer_to_device(optimizer, device)
            optimizers[module_name] = optimizer

        torch.manual_seed(random_seed)

        return TrainingState(examples_seen_so_far, modules, accumulated_modules, optimizers)

    @staticmethod
    def can_load(prefix: str,
                 module_factories: Dict[str, ModuleFactory],
                 accumulators: Dict[str, ModuleAccumulator],
                 optimizer_factories: Dict[str, OptimizerFactory]) -> bool:
        if not os.path.isdir(prefix):
            return False
        if not os.path.isfile(TrainingState.get_examples_seen_so_far_file_name(prefix)):
            return False
        for module_name in module_factories.keys():
            if not os.path.isfile(TrainingState.get_module_file_name(prefix, module_name)):
                return False
        for module_name in accumulators:
            if not os.path.isfile(TrainingState.get_accumulated_module_file_name(prefix, module_name)):
                return False
        for module_name in optimizer_factories:
            if not os.path.isfile(TrainingState.get_optimizer_file_name(prefix, module_name)):
                return False
        if not os.path.isfile(TrainingState.get_rng_state_file_name(prefix)):
            return False
        return True
