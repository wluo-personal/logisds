from logisds.data import DataLoader


class Test:
    def setup_method(self, test_method):
        self.data_loader = DataLoader(100)

    def test_data_loader_prepare_data(self):
        """ test if converted data for NN is of the right shape"""
        nn_data = self.data_loader.prepare_data()
        assert nn_data.shape[0] == 34318
        assert nn_data.shape[1] == 34

    def test_data_loader_data_gen(self):
        nn_data = self.data_loader.prepare_data()
        for each in self.data_loader.data_gen("train",batch_size=51):
            X,y = each
            break
        assert X.shape[0] == 51
        assert X.shape[1] == 100
        assert X.shape[2] == 34
        assert y.shape[0] == 51
        assert y.shape[1] == 3



