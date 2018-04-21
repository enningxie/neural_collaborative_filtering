import numpy as np
from keras import initializers
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate, Conv1D, Reshape, LSTM, add, multiply, GRU, AveragePooling1D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model_
from Dataset import Dataset
from time import time
import numpy as np
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64, 32,16,8]',
                        help="Size of each layer. Note that the first layer is the "
                             "concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--train_pd_path', type=str, default='./Data/ratings_train.csv')
    return parser.parse_args()


def get_model(num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    # user_input = Input(shape=(1,), dtype='int32', name='user_input')
    user_xz_input = Input(shape=(19,), name='user_xz_input')
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    MLP_Embedding_User_xz = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name='user_xz_embedding',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=19)
    # MLP_Embedding_User = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='user_embedding',
                                   # embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='item_embedding',
                                   embeddings_regularizer=l2(reg_layers[0]), input_length=1)

    # Crucial to flatten an embedding vector!
    conv_1 = Conv1D(kernel_size=2, filters=64, padding='same')(MLP_Embedding_User_xz(user_xz_input))
    pool_1 = AveragePooling1D()(conv_1)
    lstm_1 = LSTM(32)(pool_1)
    # user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))




    # The 0-th layer is the concatenation of embedding layers
    # vector = merge([user_latent, item_latent], mode = 'concat')
    vector = concatenate([lstm_1, item_latent])


    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', kernel_initializer=initializers.lecun_normal(),
                       name='prediction')(vector)

    model_ = Model(inputs=[user_xz_input, item_input],
                   outputs=prediction)
    print(model_.summary())
    return model_


def get_train_instances_(train, num_negatives):
    user_input, item_input, labels = [], [], []
    for u in train.keys():
        tmp_item = []
        # positive instance
        user_input.append(u)
        tmp_item.extend(train[u][:19])
        tmp_label = list(np.full(19, 1))
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # while train.has_key((u, j)):
            while j in train[u]:
                j = np.random.randint(num_items)
            tmp_item.append(j)
            tmp_label.append(0)
        item_input.append(tmp_item)
        labels.append(tmp_label)

    return user_input, item_input, labels


def get_train_instances(train, train_dict, num_negatives):
    user_input, user_xz_input, item_input, labels = [], [], [], []

    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        user_xz_input.append(train_dict[u][:19])
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            # while train.has_key((u, j)):
            while (u, j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            user_xz_input.append(train_dict[u][:19])
            item_input.append(j)
            labels.append(0)
    return user_input, user_xz_input, item_input, labels


# return a dict,
# {user_id: [movie_id1, movie_id2, movie_id3, ...]}
def get_dict_from_pd(pd_path):
    ratings_dict = dict()
    ratings_pd = pd.read_csv(pd_path)
    ratings_pd_grouped = ratings_pd.groupby(['UserID'])
    for k in ratings_pd_grouped.groups.keys():
        ratings_dict[k] = list(ratings_pd_grouped.get_group(k).sort_values(by='Timestamp')['MovieID'])
    return ratings_dict


if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    # xz add train_dict
    train_dict = get_dict_from_pd(args.train_pd_path)
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose



    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape

    # xz
    # user_input, user_xz_input, item_input, labels = get_train_instances(train, train_dict, num_negatives)
    # user_input_, item_input_, labels_ = get_train_instances_(train_dict, num_negatives)
    # print(len(user_input), len(item_input), len(labels))
    # print(len(user_input_), len(item_input_), len(labels_))
    # print(np.array(user_input).shape, np.array(user_xz_input).shape, np.array(labels).shape)
    # print(len(train.keys()))

    # -----------------------------------------------------


    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

        # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model_(model, train_dict, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, user_xz_input, item_input, labels = get_train_instances(train, train_dict, num_negatives)

        # Training
        hist = model.fit([np.array(user_xz_input), np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (hits, ndcgs) = evaluate_model_(model, train_dict, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % model_out_file)
