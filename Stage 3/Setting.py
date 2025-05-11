from local_code.base_class.setting import setting
import numpy as np

class Setting(setting):

    def load_run_save_evaluate(self):
        # 1. Load dataset → get two PyTorch DataLoaders
        loaded = self.dataset.load()
        train_loader = loaded['train_dataset']
        test_loader  = loaded['test_dataset']

        # 2. Concatenate all train batches into X_train, y_train
        Xbatches, ybatches = [], []
        for Xb, yb in train_loader:
            Xbatches.append(Xb.cpu().numpy())
            ybatches.append(yb.cpu().numpy())
        X_train = np.concatenate(Xbatches, axis=0)
        y_train = np.concatenate(ybatches, axis=0)

        # 3. Same for test set
        Xbatches, ybatches = [], []
        for Xb, yb in test_loader:
            Xbatches.append(Xb.cpu().numpy())
            ybatches.append(yb.cpu().numpy())
        X_test  = np.concatenate(Xbatches, axis=0)
        y_test  = np.concatenate(ybatches, axis=0)

        # 4. Ensure NCHW format (batch, channels, height, width).
        #    Only transpose if it looks like NHWC (i.e. last dim == 1 or 3).
        if X_train.ndim == 4 and X_train.shape[-1] in (1, 3):
            X_train = X_train.transpose(0, 3, 1, 2)
            X_test  = X_test.transpose(0, 3, 1, 2)

        # 5. Inject into method and run
        self.method.data = {
            'train': {'X': X_train, 'y': y_train},
            'test':  {'X': X_test,  'y': y_test}
        }
        learned = self.method.run()

        # 6. Save & evaluate
        self.result.data   = learned
        self.result.save()
        self.evaluate.data = learned
        return self.evaluate.evaluate()
