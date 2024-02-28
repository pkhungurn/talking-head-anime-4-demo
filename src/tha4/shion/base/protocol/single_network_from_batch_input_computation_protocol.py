from typing import Optional, Any, List

from tha4.shion.core.cached_computation import CachedComputationProtocol, ComputationState

KEY_NETWORK = "network"
KEY_NETWORK_OUTPUT = "network_output"


class SingleNetworkBatchInputComputationProtocol(CachedComputationProtocol):
    def __init__(self,
                 key_network: str = KEY_NETWORK,
                 key_network_output: str = KEY_NETWORK_OUTPUT,
                 input_index_to_batch_index: Optional[List[int]] = None):
        if input_index_to_batch_index is None:
            input_index_to_batch_index = [0]

        self.input_index_to_batch_index = input_index_to_batch_index
        self.key_network_output = key_network_output
        self.key_network = key_network

    def compute_output(self, key: str, state: ComputationState) -> Any:
        if key == self.key_network_output:
            inputs = []
            for batch_index in self.input_index_to_batch_index:
                inputs.append(state.batch[batch_index])
            network = state.modules[self.key_network]
            return network.forward(*inputs)
        else:
            raise RuntimeError("Computing output for key " + key + " is not supported!")
