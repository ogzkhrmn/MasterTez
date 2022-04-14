import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_data(filename):
    mylist = []

    for chunk in pd.read_csv(filename, index_col='date', parse_dates=True, low_memory=False, chunksize=20000):
        mylist.append(chunk)

    data = pd.concat(mylist, axis=0)
    del mylist


    data["bought"].replace({"f": 0, "t": 1}, inplace=True)

    # Drop unnecessary columns
    data = data.drop(columns={'id', 'detectedcorruption',
                              'detectedduplicate', 'location', 'n', 'q', 'referer', 'remotehost', 'useragenttype',
                              'useragentvendor', 'city', 'country', 'path_name', 'ready', 'sent', 'localpath',
                              'partyid', 'staytime', 'useragentdevicecategory', 'useragentfamily',
                              'useragentname', 'useragentosfamily', 'useragentosvendor', 'useragentosversion',
                              'useragentstring', 'long_time', 'useragentversion', 'timestamp'})

    # Change string values to categorical values
    data.utm_campaign = pd.Categorical(pd.factorize(data.utm_campaign)[0])
    data.utm_medium = pd.Categorical(pd.factorize(data.utm_medium)[0])
    data.utm_content = pd.Categorical(pd.factorize(data.utm_content)[0])
    data.utm_source = pd.Categorical(pd.factorize(data.utm_source)[0])
    data.eventtype = pd.Categorical(pd.factorize(data.eventtype)[0])
    data.firstinsession = pd.Categorical(pd.factorize(data.firstinsession)[0])
    data.mid = pd.Categorical(pd.factorize(data.mid)[0])
    data.mouseevent = pd.Categorical(pd.factorize(data.mouseevent)[0])
    data.pageviewid = pd.Categorical(pd.factorize(data.pageviewid)[0])
    data.sessionid = pd.Categorical(pd.factorize(data.sessionid)[0])
    data.userip = pd.Categorical(pd.factorize(data.userip)[0])

    # One Hot encoding for selected columns
    # dum_df = pd.get_dummies(data, columns=["city", "country"])
    # data = pd.concat([data, dum_df], axis=1)

    # Standardize features
    data[["viewportpixelheight", "viewportpixelwidth", "screenx", "screeny", "pagex", "pagey", "screenpixelheight",
          "screenpixelwidth"]] = StandardScaler().fit_transform(data[[
        "viewportpixelheight", "viewportpixelwidth", "screenx", "screeny", "pagex", "pagey", "screenpixelheight",
        "screenpixelwidth"]])

    data.insert(len(data.columns) - 1, 'bought', data.pop('bought'))

    data.fillna(data.median(numeric_only=True), inplace=True)
    return data


def get_data(data_frame, n_inputs, channel_size):
    y = data_frame.iloc[:, (n_inputs - channel_size):n_inputs]
    x = data_frame.iloc[:, :-1 * channel_size]
    ss = StandardScaler()
    x_ss = ss.fit_transform(x)
    return x_ss, y
