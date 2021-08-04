from logisds.model import TSModel

class Test:
    def setup_method(self, test_method):
        self.model = TSModel(N_features=3,N_steps=10,lr=0.01)

    def test_get_model(self):
        """ test if converted data for NN is of the right shape"""
        model = self.model.get_model()
        # test input
        assert model.input.shape[0] is None
        assert model.input.shape[1] == 10
        assert model.input.shape[2] == 3

        # test output
        assert model.output.shape[0] is None
        assert model.output.shape[1] == 3

