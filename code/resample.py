



def load_cubic_interp_pandas():
    """
    Note: pandas cubic spline has its problem: dataframe.interpolate(method='spline', order=3)
	                             time_tick  acc_X_value  acc_Y_value  acc_Z_value
	time                                                                     
	1970-01-01 00:01:10.006     70.006      0.00853    -0.061354     9.785901
	1970-01-01 00:01:10.007        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.008        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.009        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.010        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.011        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.012        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.013        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.014        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.015        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.016     70.016     -0.00015     0.025140     9.785751
	1970-01-01 00:01:10.017        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.018        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.019        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.020        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.021        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.022        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.023        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.024        NaN          NaN          NaN          NaN
	1970-01-01 00:01:10.025        NaN          NaN          NaN          NaN
	                         time_tick  acc_X_value  acc_Y_value  acc_Z_value
	time                                                                     
	1970-01-01 00:01:10.006     70.006     0.008530    -0.061354     9.785901
	1970-01-01 00:01:10.007     70.007     0.021214     0.004320     9.786631
	1970-01-01 00:01:10.008     70.008     0.021216     0.004319     9.786631
	1970-01-01 00:01:10.009     70.009     0.021218     0.004319     9.786631
	1970-01-01 00:01:10.010     70.010     0.021220     0.004319     9.786631
	1970-01-01 00:01:10.011     70.011     0.021222     0.004319     9.786632
	1970-01-01 00:01:10.012     70.012     0.021224     0.004319     9.786632
	1970-01-01 00:01:10.013     70.013     0.021226     0.004319     9.786632
	1970-01-01 00:01:10.014     70.014     0.021228     0.004319     9.786632
	1970-01-01 00:01:10.015     70.015     0.021230     0.004318     9.786632
	1970-01-01 00:01:10.016     70.016    -0.000150     0.025140     9.785751
	1970-01-01 00:01:10.017     70.017     0.021233     0.004318     9.786633
	1970-01-01 00:01:10.018     70.018     0.021235     0.004318     9.786633
	1970-01-01 00:01:10.019     70.019     0.021237     0.004318     9.786633
	1970-01-01 00:01:10.020     70.020     0.021239     0.004318     9.786633
	1970-01-01 00:01:10.021     70.021     0.021241     0.004318     9.786634
	1970-01-01 00:01:10.022     70.022     0.021243     0.004317     9.786634
	1970-01-01 00:01:10.023     70.023     0.021245     0.004317     9.786634
	1970-01-01 00:01:10.024     70.024     0.021247     0.004317     9.786634
	1970-01-01 00:01:10.025     70.025     0.021249     0.004317     9.786635
    input: 
        -- df
    output:
        -- df_resample

    """
    df = pd.read_csv('../test/lemon41gbound-acc.csv')
    df = df[(df['time_tick']<90) & (df['time_tick']>70)]

    # df = df[['time', col]]
    df['time_tick'] = (df['time_tick'].values*1000).astype(int)/1000
    df['time'] = pd.to_datetime(df['time_tick'], unit='s')
    df = df.set_index('time')
    df_resample = df.resample('0.001S').asfreq()
    print(df_resample.head(20))
    df_resample['time_tick'] = df_resample['time_tick'].interpolate(method='spline', order=3)
    df_resample['acc_X_value'] = df_resample['acc_X_value'].interpolate(method='spline', order=3)
    df_resample['acc_Y_value'] = df_resample['acc_Y_value'].interpolate(method='spline', order=3)
    df_resample['acc_Z_value'] = df_resample['acc_Z_value'].interpolate(method='spline', order=3)
    print(df_resample.head(20))
    # df_resample = df_resample.dropna()
    t = df_resample['time_tick'].values
    x = df_resample['acc_X_value'].values
    y = df_resample['acc_Y_value'].values
    z = df_resample['acc_Z_value'].values    
    plt_raw_xyz(t, x, y, z)

    return df_resample
