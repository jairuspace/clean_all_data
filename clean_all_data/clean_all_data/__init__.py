def clean_all_data(training_data, predicting_data, output_file, predict_file, drop_cols, id_col, response_var, sample_2_prop):

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    #Read in Data
    train_frame = pd.read_csv(training_data)
    train_len = len(train_frame)+1

    predict_frame = pd.read_csv(predicting_data)
    predict_len = -len(predict_frame)

    dataframe = pd.concat([train_frame,predict_frame])
    dataframe = dataframe.drop(drop_cols, axis=1)

    #Clean Data
    coded_frame = pd.DataFrame()
    i = 0
    for column in dataframe:
        if str(dataframe.columns.values[i]) == str(id_col):
            coded_frame[column] = list(dataframe[column])
            i += 1
            continue
        else:
            if dataframe[column].dtype == 'object':
                dataframe[column] = dataframe[column].fillna('notfound')
            else:
                dataframe[column] = dataframe[column].fillna(0)
            # i += 1
            le = LabelEncoder()
            le.fit(dataframe[column])
            transformed = le.transform(dataframe[column])
            coded_frame[column] = transformed
        i += 1

    train_frame = coded_frame[:train_len]
    predict_frame = coded_frame[predict_len:]

    dataframe = train_frame

    dataframe = dataframe.sample(frac=1)

    if sample_2_prop is True:
        # Downsample to equal proportions
        complete = dataframe[dataframe[response_var] == 1]
        incomplete = dataframe[dataframe[response_var] == 0]
        if len(complete) < len(incomplete):
            incomplete = incomplete.sample(len(complete))
        elif len(complete) > len(incomplete):
            complete = complete.sample(len(incomplete))
        dataframe = pd.concat([complete,incomplete])
        dataframe = dataframe.sample(frac=1)

    #Save to CSV
    dataframe.to_csv(output_file)
    predict_frame.to_csv(predict_file)