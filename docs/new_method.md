# Integrate a new method

Suppose you generate a new synthetic dataset named <em>**new_sota.pt**</em> using <em>**IPC1**</em> using your new method: <em>**awesome_method**</em>, please follow the steps below to integrate them into DC-Bench.

## Step 1: Add synthetic data
All the synthetic dataset are stored under distilled_results directory with the strucute being
- distilled_results
  - <em>**awesome_method**</em>
    - <em>**IPC1**</em>
      - <em>**new_sota.pt**</em>

## Step 2: Implement a dataset loader(Optional)
Most of the data loader can be **reused** if they are PyTorch tensors or numpy arrays. Here is the API we defined for loading the synthetic datasets(demonstrated using DC method)
```
class DCDataLoader:
    @staticmethod
    def load_data(root_dir, dataset, ipc, data_file):
        data_path = os.path.join(root_dir, "DC", dataset, 'IPC' + str(ipc), data_file)
        dc_data = torch.load(data_path)
        training_data = dc_data['data']
        train_images, train_labels = training_data[-1]
        return train_images, train_labels
```
The return results are two PyTorch tensors containing the training images and training labels.
