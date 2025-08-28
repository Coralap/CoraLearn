class Optimizer:
    def update(self, layer=None, layers=None, X=None, y=None, loss=None, layer_input=None):
        raise NotImplementedError
    @property
    def network_level(self):
        # By default, per-layer
        return False
