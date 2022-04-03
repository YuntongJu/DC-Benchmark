class EvaluatorUtils:

    def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args):
        net = net.to(args.device)
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss().to(args.device)

        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

        start = time.time()
        for ep in range(Epoch+1):
            loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        time_train = time.time() - start
        loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

        return net, acc_train, acc_test

    def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
        loss_avg, acc_avg, num_exp = 0, 0, 0
        net = net.to(args.device)
        criterion = criterion.to(args.device)

        if mode == 'train':
            net.train()
        else:
            net.eval()

        for i_batch, datum in enumerate(dataloader):
            img = datum[0].float().to(args.device)
            if aug:
                if args.dsa:
                    img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
                else:
                    img = augment(img, args.dc_aug_param, device=args.device)
            lab = datum[1].long().to(args.device)
            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        loss_avg /= num_exp
        acc_avg /= num_exp

        return loss_avg, 
        