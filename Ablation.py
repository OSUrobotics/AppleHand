#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:13:18 2021

@author: orochi
"""


def perform_ablation(reduced_train_state, reduced_test_state, reduced_train_label, reduced_test_label):
    labels = {'IMU Accelearation': [0, 1, 2, 9, 10, 11, 18, 19, 20], 'IMU Velocity': [3, 4, 5, 12, 13, 14, 21, 22, 23],
              'Joint Pos': [6, 15, 24],
              'Joint Velocity': [7, 16, 25], 'Joint Effort': [8, 17, 26], 'Arm Joint State': [27, 28, 29, 30, 31, 32],
              'Arm Joint Velocity': [33, 34, 35, 36, 37, 38], 'Arm Joint Effort': [39, 40, 41, 42, 43, 44],
              'Wrench Force': [45, 46, 47],
              'Wrench Torque': [48, 49, 50]}
    full_list = np.array(range(51))
    missing_labels = []
    missing_names = ''
    worst_names = []
    sizes = [51]
    reduced_train_data_grasp_success = TensorDataset(torch.from_numpy(reduced_train_state),
                                                     torch.from_numpy(reduced_train_label[:, -2]))
    reduced_test_data_grasp_success = TensorDataset(torch.from_numpy(reduced_test_state),
                                                    torch.from_numpy(reduced_test_label[:, -2]))
    reduced_train_loader_grasp_success = DataLoader(reduced_train_data_grasp_success, shuffle=False,
                                                    batch_size=batch_size, drop_last=True)
    reduced_test_loader_grasp_success = DataLoader(reduced_test_data_grasp_success, shuffle=False,
                                                   batch_size=batch_size, drop_last=True)
    performance = []
    for i in range(3):
        trained_grasp_lstm, best_accuracies, losses, steps, TP_rate, FP_rate = train(reduced_train_loader_grasp_success,
                                                                                     reduced_test_loader_grasp_success,
                                                                                     len(reduced_test_state), epochs=60,
                                                                                     model_type='LSTM', output='grasp',
                                                                                     train_points=len(
                                                                                         reduced_train_state))
        performance = np.max(best_accuracies)
        print('model finished, saving now')
        torch.save(trained_grasp_lstm, 'grasp_lstm_all' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pt')
        grasp_lstm_dict = {'acc': best_accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
        file = open('grasp_lstm_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
        pkl.dump(grasp_lstm_dict, file)
    file.close()
    best_accuracies = [np.average(performance)]
    best_acc_std_dev = np.std(best_accuracies)
    print(best_accuracies)
    for phase in range(9):
        big_accuracies = []
        names = []
        indexes = []
        acc_std_dev = []
        for name, missing_label in labels.items():
            temp = np.ones(51, dtype=bool)
            temp[missing_label] = False
            try:
                temp[missing_labels] = False
            except:
                pass
            used_labels = full_list[temp]
            reduced_train_data_grasp_success = TensorDataset(torch.from_numpy(reduced_train_state[:, used_labels]),
                                                             torch.from_numpy(reduced_train_label[:, -2]))
            reduced_test_data_grasp_success = TensorDataset(torch.from_numpy(reduced_test_state[:, used_labels]),
                                                            torch.from_numpy(reduced_test_label[:, -2]))
            reduced_train_loader_grasp_success = DataLoader(reduced_train_data_grasp_success, shuffle=False,
                                                            batch_size=batch_size, drop_last=True)
            reduced_test_loader_grasp_success = DataLoader(reduced_test_data_grasp_success, shuffle=False,
                                                           batch_size=batch_size, drop_last=True)
            performance = []
            for i in range(3):
                trained_grasp_lstm, accuracies, losses, steps, TP_rate, FP_rate = train(
                    reduced_train_loader_grasp_success, reduced_test_loader_grasp_success, len(reduced_test_state),
                    epochs=60, model_type='LSTM', output='grasp', input_dim=len(reduced_train_state[0, used_labels]))
                performance.append(np.max(accuracies))
                print('model finished, saving now')
                torch.save(trained_grasp_lstm, 'grasp_lstm_' + missing_names + name + datetime.datetime.now().strftime(
                    "%m_%d_%y_%H%M") + '.pt')
                grasp_lstm_dict = {'acc': accuracies, 'loss': losses, 'steps': steps, 'TP': TP_rate, 'FP': FP_rate}
                file = open('grasp_lstm_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
                pkl.dump(grasp_lstm_dict, file)
                file.close()
            big_accuracies.append(np.average(performance))
            acc_std_dev.append(np.std(performance))
            names.append(name)
            indexes.append(missing_label)
        best_one = np.argmax(big_accuracies)
        print('best thing', best_one, np.shape(big_accuracies), np.shape(names))
        missing_names = missing_names + names[best_one]
        missing_labels.extend(indexes[best_one])
        best_accuracies.append(big_accuracies[best_one])
        best_acc_std_dev.append(acc_std_dev[best_one])
        worst_names.append(names[best_one])
        sizes.append(sizes[phase] - len(labels[names[best_one]]))
        labels.pop(names[best_one])
    print('ablation finished. best accuracies throughout were', best_accuracies)
    print('names removed in this order', worst_names)

    grasp_ablation_dict = {'num inputs': sizes, 'best accuracy': best_accuracies, 'std dev': best_acc_std_dev,
                           'names': worst_names}
    file = open('grasp_ablation_data' + datetime.datetime.now().strftime("%m_%d_%y_%H%M") + '.pkl', 'wb')
    pkl.dump(grasp_ablation_dict, file)
    file.close()

