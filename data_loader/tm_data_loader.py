import torch

class TMDataLoader:

    @staticmethod
    def load_data(dataset, ipc, image_path, label_path):
        if dataset == 'tinyimagenet' and ipc == 50:
            # the data are broken down into 5 files.
            images = []
            labels = []
            for i in range(5):
                images.append(torch.load(image_path + '_' + str(i) + '.pt'))
                labels.append(torch.load(label_path + '_' + str(i) + '.pt'))
            return torch.cat(images), torch.cat(labels)
        else:
            return torch.load(image_path), torch.load(label_path)


if __name__ == '__main__':
    data_path = '/nfs/data/justincui/dc_benchmark/distilled_results/TM/tinyimagenet/IPC50/'
    images, labels = TMDataLoader.load_data("tinyimagenet", 50, data_path + 'images_best', data_path + 'labels_best')
    print(images.shape)
    print(labels.shape)