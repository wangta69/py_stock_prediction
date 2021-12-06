import numpy as np
from keras.models import load_model
from util import csv_to_dataset, multiple_csv_to_dataset, history_points
import matplotlib.pyplot as plt

model = load_model('technical_model.h5')

ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = csv_to_dataset('MSFT_daily.csv')
# ohlcv_histories, technical_indicators, next_day_open_values, unscaled_y, y_normaliser = multiple_csv_to_dataset('MSFT_daily.csv')
# print(ohlcv_histories)
# print(technical_indicators)
#

test_split = 0.9
n = int(ohlcv_histories.shape[0] * test_split)

ohlcv_train = ohlcv_histories[:n]
tech_ind_train = technical_indicators[:n]
y_train = next_day_open_values[:n]

ohlcv_test = ohlcv_histories[n:]
tech_ind_test = technical_indicators[n:]
y_test = next_day_open_values[n:]

unscaled_y_test = unscaled_y[n:]

y_test_predicted = model.predict([ohlcv_test, tech_ind_test])
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)

buys = []
sells = []
thresh = 0.1

start = 0
end = -1

x = -1

# print('ohlcv_test ========================')
# print(ohlcv_test)
# print('tech_ind_test ========================')
# print(tech_ind_test)

for ohlcv, ind in zip(ohlcv_test[start: end], tech_ind_test[start: end]):
    print('===================================')
    print(ohlcv, ind)
    print('+++++++++++++++++++++++++++++++++++')

    normalised_price_today = ohlcv[-1][0]
    normalised_price_today = np.array([[normalised_price_today]])
    price_today = y_normaliser.inverse_transform(normalised_price_today)
    print('1 Shape of ohlcv is ', ohlcv.shape)
    print('1 Shape of ind is', ind.shape)
    # ind = ind.reshape(1, -1) # y = y.reshape(1,-1), which makes the First Dimension (Batch_Size) equal (1) for both X and y.
    # ind = ind.reshape(1, -1)  # y = y.reshape(1,-1), which makes the First Dimension (Batch_Size) equal (1) for both X and y.
    # ind = np.reshape(ind, (5, 50))  # y = y.reshape(1,-1), which makes the First Dimension (Batch_Size) equal (1) for both X and y.
    # ind = np.reshape(ind, (ind.shape[0], ind.shape[0]), 1)
    # ohlcv = np.reshape(ohlcv, (ohlcv.shape[0], ohlcv.shape[0]), 1)
    # ind = np.reshape(ind, (ind.shape[0], ind.shape[0]), 1)
    print('2 Shape of ohlcv is ', ohlcv.shape)
    print('2 Shape of ind is', ind.shape)
    # print(model.predict([[ohlcv], [ind]]))

    exit()
    # predicted_price_tomorrow = np.squeeze(y_normaliser.inverse_transform(model.predict([[ohlcv], [ind]])))
    # delta = predicted_price_tomorrow - price_today
    # if delta > thresh:
    #     buys.append((x, price_today[0][0]))
    # elif delta < -thresh:
    #     sells.append((x, price_today[0][0]))
    # x += 1


print(f"buys: {len(buys)}")
print(f"sells: {len(sells)}")


def compute_earnings(buys_, sells_):
    purchase_amt = 10
    stock = 0
    balance = 0
    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys_[0][1]
            buys_.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop(0)
    print(f"earnings: ${balance}")


# we create new lists so we dont modify the original
compute_earnings([b for b in buys], [s for s in sells])



plt.gcf().set_size_inches(22, 15, forward=True)

real = plt.plot(unscaled_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')

if len(buys) > 0:
    plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
if len(sells) > 0:
    plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')

plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])

plt.show()
