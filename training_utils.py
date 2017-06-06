import time
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output


def get_pred(y):
    digits = np.argmax(y, axis=-1)
    return [''.join([str(x).replace('10', '_') for x in dig]) for dig in digits]


def accuracy(y_true, y_pred):
    y_true = y_true
    y_pred = y_pred
    actual = np.array(get_pred(y_true))
    pred = np.array(get_pred(y_pred))
    return np.mean(actual == pred)


def capture_performance2(model, train_x, train_y, valid_x, valid_y, batch_size, batch_idx,
                         compare_every, t_acc, v_acc, verbose=0):
    #     print('Saving performance')
    try:
        train_dig_proba = model.predict(
            train_x[batch_size * (batch_idx - compare_every + 1):batch_size * (batch_idx + 1)])
        t_acc.append(accuracy(train_y[batch_size * (batch_idx - compare_every + 1):batch_size * (batch_idx + 1)],
                              train_dig_proba))
    except:
        t_acc.append(np.nan)

    valid_dig_proba = model.predict(valid_x)
    v_acc.append(accuracy(valid_y, valid_dig_proba))
    if verbose > 0:
        print('capture_performance2(): valid accuracy {}'.format(v_acc[-1]))


def train_model(model, train_x, train_y, valid_x, valid_y, n_epochs=20, batch_size=20, max_steps_per_epoch=100000000,
                compare_every=100):
    t_acc = []
    v_acc = []

    def log_acc(batch_index):
        capture_performance2(model, train_x, train_y, valid_x, valid_y, batch_size,
                             batch_index, compare_every, t_acc, v_acc, verbose=1)

    np.random.seed(123456)
    t1 = time.time()
    for ep in range(n_epochs):
        s1 = time.time()
        idx = np.arange(len(train_x))
        np.random.shuffle(idx)  # shuffling the data each epoch
        train_x = train_x[idx]
        train_y = train_y[idx]
        ep_dig_losses = []
        for batch_idx in range(min(len(train_x) // batch_size, max_steps_per_epoch)):
            batch_x = train_x[batch_size * batch_idx:(batch_size * (batch_idx + 1))]
            batch_y = train_y[batch_size * batch_idx:(batch_size * (batch_idx + 1))]
            total_loss = model.train_on_batch(batch_x, batch_y)

            ep_dig_losses.append(total_loss)
            if batch_idx % compare_every == 0:
                log_acc(batch_idx)
        clear_output()
        print('{:4.2f} minutes for epoch'.format((time.time() - s1) / 60))
        log_acc(batch_idx)

        v_dig_proba = model.predict(valid_x)
        v_dig_loss = model.evaluate(valid_x, valid_y, verbose=0)
        print("Epoch {}".format(ep))
        #     print(v_acc)
        print("Validation accuracy {:<2.3%}".format(accuracy(valid_y, v_dig_proba)))

        print("Mean batch dig loss {:<2.3f}".format(np.mean(ep_dig_losses)))
        print("Valid dig loss {:<2.3f}".format(np.mean(v_dig_loss)))

        plt.plot(t_acc, label='Train Accuracy')
        plt.plot(v_acc, label='Validation Accuracy')
        plt.ylim((0, 1))
        plt.legend(loc='best')
        plt.show()
    print('{:4.2f} minutes total elapsed'.format((time.time() - t1) / 60))

    return model, t_acc, v_acc