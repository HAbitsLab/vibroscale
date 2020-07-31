import glob,os
import scipy.fftpack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statistics

from resample import resample as linear_interpl


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def pca_sensor_xyz_xy(df):
    pca = PCA(n_components=1)
    df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']] -= df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].mean()
    df['acc_xyz_pca'] = pca.fit_transform( df[['acc_X_value', 'acc_Y_value', 'acc_Z_value']].to_numpy())
    df['acc_xy_pca'] = pca.fit_transform( df[['acc_X_value', 'acc_Y_value']].to_numpy())
    return df


def resample_df(df):
    df = linear_interpl(df, time_col_header='time_tick', sampling_rate=200)
    return df


def fruit(path, delay, starttime):
    df = pd.read_csv(path)
    df = resample_df(df)
    df = df[(df['time_tick']>(starttime + delay + 6500))&(df['time_tick']<(starttime + delay + 7500))]
    df = pca_sensor_xyz_xy(df)
    t = df['time_tick'].values
    t = (t - t[0])/1000
    x = df['acc_X_value'].values
    y = df['acc_Y_value'].values
    z = df['acc_Z_value'].values
    xyz_pca = df['acc_xyz_pca'].values
    xy_pca = df['acc_xy_pca'].values
    return t, x, y, z, xyz_pca, xy_pca


def fruit_no_load(path, delay, starttime):
    df = pd.read_csv(path)
    df = resample_df(df)
    df = df[(df['time_tick']>(starttime + 4000))&(df['time_tick']<(starttime + 5000))]
    df = pca_sensor_xyz_xy(df)
    t = df['time_tick'].values
    t = (t - t[0])/1000
    x = df['acc_X_value'].values
    y = df['acc_Y_value'].values
    z = df['acc_Z_value'].values
    xyz_pca = df['acc_xyz_pca'].values
    xy_pca = df['acc_xy_pca'].values
    return t, x, y, z, xyz_pca, xy_pca


if __name__ == '__main__':
    # ===============================================================
    script_path = os.path.dirname(os.path.realpath(__file__))

    apple_path = os.path.join(script_path, '../Data/7.2_nexus2_rotate/apple')
    tableware_path = os.path.join(script_path, '../Data/7.2_nexus2_rotate/tools')
    onion_path = os.path.join(script_path, '../Data/7.4_nexus2_rotate/onion')
    pepper_path = os.path.join(script_path, '../Data/7.4_nexus2_rotate/pepper')
    # ===============================================================
    paths = [apple_path, onion_path, pepper_path, tableware_path]
    weights_all = []
    intensity_load_all = []
    intensity_no_load_all = []

    for path in paths:
        # validation is 'LOOCV'
        # TODO: can use function to get the info from counting files
        if path == apple_path:
            n_splits = 24
        elif path == tableware_path:
            n_splits = 6
        elif path == pepper_path:
            n_splits = 6
        elif path == onion_path:
            n_splits = 16
        else:
            print('error!')
            exit()

        os.chdir(path)
        print('path:', path)
        print('n_splits:', n_splits)

        weights = []
        intensity_load = []
        intensity_no_load = []
        for counter, current_file in enumerate(glob.glob("*.csv")):
            param = current_file.split('_')
            """
            Param index: SAPPLE_1_125_1593191386730_45000_0.8889_2
            0: Fruit name
            1: ID
            2: weight
            3: start time
            4: cycle
            5: delay
            6: ratio
            7: repeat
            """
            tf, xf, yf, zf, xyz_pcaf, xy_pcaf = fruit(current_file,int(param[4]),int(param[3]))
            weights.append(int(param[2]))

            te, xe, ye, ze, xyz_pcae, xy_pcae = fruit_no_load(current_file,int(param[4]),int(param[3]))
            # ======================================
            # y-axis: MAE = 12.7
            # ========================================
            yf -= np.mean(yf)
            intense_f = np.mean(np.abs(yf))
            # print("{} {} {} {}:{:.6f}\n".format(param[0],param[1],param[2], param[3],intense))
            intensity_load.append(intense_f)
            ye -= np.mean(ye)
            intense_e = np.mean(np.abs(ye))
            intensity_no_load.append(intense_e)


        intensity_net = np.asarray(intensity_no_load) - np.asarray(intensity_load)
        data = pd.DataFrame({'weights': weights, 'intensity_net': intensity_net})
        data = data.sort_values('intensity_net')
        X = data['intensity_net'].values
        y = data['weights'].values
        kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
        kf.get_n_splits(X, y)

        final_GT_list = []
        final_pred_list = []

        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index].reshape(-1, 1), X[test_index].reshape(-1, 1)
            y_train, y_test = y[train_index], y[test_index]
            reg = LinearRegression().fit(X_train, y_train)
            pred = reg.predict(X_test)
            final_GT_list.append(y_test)
            final_pred_list.append(pred)

        pred = np.hstack(final_pred_list)
        gt = np.hstack(final_GT_list)
        print('total number:', len(gt))
        print('min/max weight (GT):', np.min(gt), np.max(gt))
        rmse_val = rmse(pred, gt)
        print("MAE (mean absolute error) is: ", mean_absolute_error(pred, gt))
        print("rms error is: " + str(rmse_val) + '\n\n')

        # standard deviation for absolute error
        print('gt', gt)
        print('pred', pred)
        print(np.abs(gt-pred))
        print("standard deviation for absolute error", statistics.stdev(np.abs(gt-pred)))

        weights_all += weights
        intensity_load_all += intensity_load
        intensity_no_load_all += intensity_no_load

    print(len(weights_all))
    print(len(intensity_load_all))
    print(len(intensity_no_load_all))

    intensity_net_all = np.asarray(intensity_no_load_all) - np.asarray(intensity_load_all)
    data_all = pd.DataFrame({'weights': weights_all, 'intensity_net': intensity_net_all})
    data_all = data_all.sort_values('intensity_net')
    X = data_all['intensity_net'].values
    y = data_all['weights'].values
    kf = KFold(n_splits=len(weights_all), random_state=1, shuffle=True)
    kf.get_n_splits(X, y)
    final_GT_list = []
    final_pred_list = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index].reshape(-1, 1), X[test_index].reshape(-1, 1)
        y_train, y_test = y[train_index], y[test_index]
        reg = LinearRegression().fit(X_train, y_train)
        # print(reg.score(X_train, y_train))
        # print(reg.coef_)
        pred = reg.predict(X_test)
        final_GT_list.append(y_test)
        final_pred_list.append(pred)


    pred = np.hstack(final_pred_list)
    gt = np.hstack(final_GT_list)
    # print('average weights (GT):', np.mean(gt))
    print('objects:',paths)
    print('total number:', len(gt))
    print('min/max weight (GT):', np.min(gt), np.max(gt))
    rmse_val = rmse(pred, gt)
    print("MAE (mean absolute error) is: ", mean_absolute_error(pred, gt))
    print("rms error is: " + str(rmse_val))

    plt.scatter(weights_all, intensity_net_all)
    # plt.show()
    plt.savefig(os.path.join(script_path, "../Figure/intensity_vs_weight.eps"), format='eps')


