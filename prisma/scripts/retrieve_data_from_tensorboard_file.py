#!/usr/bin python3
# -*- coding: utf-8 -*-
""" 
First, this script will parse the data from the tensorboard files found in the provided directory and store the result in a Dataframe object.
Then, it will go through the experiements folder and compare the results of each experience (test global cost) to SP, and so filter the experiments to good and bad given the parameters used.

sources : 
- https://laszukdawid.com/blog/2021/01/26/parsing-tensorboard-data-locally/
- https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file
"""
import tkinter
import matplotlib
matplotlib.use( 'TkAgg' )

def convert_tb_data(root_dir, sort_by=None):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator
    from tensorflow.python.framework import tensor_util

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        if "hparams" in tfevent.summary.value[0].tag:
            scalar = 0.0
        else:
            scalar = tensor_util.MakeNdarray(tfevent.summary.value[0].tensor).item()
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=scalar,
        )
    
    columns_order = ['wall_time', 'name', 'step', 'value']
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out)[columns_order]
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns 
    
    # Define the exps folder and evaluation metric name
    dir_path = "/home/redha/PRISMA/prisma/examples/abilene/5_nodes_full_mesh_overlay_experiment/to_test"
    eval_metric_name = "test_global_cost"
    # Retrive the SP results
    sp_df = convert_tb_data(f"{dir_path}/sp")
    sp_values = np.array(sp_df[sp_df.name == eval_metric_name].sort_values("step").value.values, dtype=float)
    
    # Retrive the SP results
    opt_df = convert_tb_data(f"{dir_path}/opt")
    opt_values = np.array(opt_df[opt_df.name == eval_metric_name].sort_values("step").value.values, dtype=float)
    
    # go through the folder and retrieve the results for each experiment
    folders_to_skip = ["saved_models", "good", "bad", "to_test", "a_test", "sp", "opt"]
    all_exps_results = []
    for exp_name in os.listdir(dir_path):
        
        # Skip the unrelevant folders 
        if exp_name in folders_to_skip:
            continue
        
        # Get the results for the experiment
        exp_df = convert_tb_data(f"{dir_path}/{exp_name}")
        exp_values = np.array(exp_df[exp_df.name == eval_metric_name].sort_values("step").value.values, dtype=float)
        
        # Skip instances that don't have test results
        if len(exp_values) == 0:
            continue
        
        # Check if the results are better than SP 
        exp_flag = int(np.all(exp_values[2:5] < sp_values[2:5]))
        
        # Compute the MSE between the exp and the opt solution
        exp_mse = np.mean(np.abs((exp_values[2:5] - opt_values[2:5])/opt_values[2:5]))
        
        # Write everything to a dict
        exp_dict = dict(
            flag=exp_flag,
            mse=exp_mse,
            exp_values=exp_values
        )
        
        # Add the parameters of the experiment
        params_names = ["ping frequency", "train load", "use throughput",  "moving average window", "learning rate", "batch size", "exploration fixed"]
        params_markers = ["freq", "load", "dqn", "avg", "lr", "bs", "explo"]
        name_splitted = exp_name.split("_")
        for i, param_marker in enumerate(params_markers):
            param_name = params_names[i]
            param_value =  name_splitted[name_splitted.index(param_marker)+1]
            if param_name == "use throughput":
                if param_value == "with":
                    param_value = 1
                else:
                    param_value = 0
            if param_name == "exploration fixed":
                if param_value == "fixed":
                    param_value = 1
                else:
                    param_value = 0
            exp_dict[param_name] = float(param_value)
            
        # Append to the dataframe
        all_exps_results.append(pd.DataFrame(exp_dict))
        
    # Sort and fix the variables
    all_exps_df = pd.concat(all_exps_results)[params_names+["flag", "mse"]]
    all_exps_df.reset_index()
    print(all_exps_df)
    
    # Analyze the results
    # pd.plotting.scatter_matrix(all_exps_df,figsize=(20,15))
    
    #%%
    # for param_name in params_names:
    #     plt.figure(figsize=(16,7))
    #     sns.scatterplot(x=param_name, y="mse", data=all_exps_df, hue="flag")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale

    
    # Data to treat 
    data = scale(all_exps_df[params_names+["mse"]].values)
    
    # check pca
    nb_components = 6
    pca = PCA(n_components=nb_components)
    pca_result = pca.fit_transform(data)
    components_names = []
    for i in  range(nb_components):
        all_exps_df[f'pca-{i}'] = pca_result[:,i]
        components_names.append(f'pca-{i}')
    


    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        
    # Check the correlation between the principal components and the MSE values
    print(all_exps_df[components_names + ["mse"]].corr()["mse"])
    
    # plot the most correlated component 
    # plt.figure(figsize=(16,7))
    # # for i in range(1, nb_components+1):
    #     # ax = plt.subplot(int(nb_components/np.floor(np.sqrt(nb_components))),
    #     #             int(np.floor(np.sqrt(nb_components))),
    #     #             i)
    # sns.scatterplot(
    #     x="pca-0",y="mse",
    #     hue="flag",
    #     data=all_exps_df,
    #     alpha=0.5,
    # )

    
    
    # print the coefficient of the most correlated componenent
    for i,x in enumerate(params_names+["mse"]):
        print(x, pca.components_[0][i])
    print("done")
    
    #%%
    # Trying MCA
    # params_names.remove('use throughput')
    import prince
    mca = prince.MCA(n_components=nb_components,
                    copy=True,
                    check_input=True,
                    engine='auto',
                    random_state=42)
    X = all_exps_df[params_names + ["flag"]]
    for column in X.columns:
        X[column] = X[column].apply(str)
    # from sklearn.preprocessing import OneHotEncoder
    # encoder = OneHotEncoder(handle_unknown='ignore')

    # X = pd.DataFrame(encoder.fit_transform(all_exps_df[params_names]).toarray()) 
    # columns = []
    # for name in params_names:
    #     columns.extend([f"{name}-{x}" for x in all_exps_df[name].unique().tolist()])
    # X.columns = columns
    mca_components_names = []
    mca_result = mca.fit_transform(X).values
    for i in  range(nb_components):
        all_exps_df[f'mca-{i}'] = mca_result[:,i]
        mca_components_names.append(f'mca-{i}')
    
    print(all_exps_df[mca_components_names + ["mse"]].corr()["mse"])

    # ax = mca.plot_coordinates(
    #     X=X,
    #     ax=None,
    #     figsize=(6, 6),
    #     show_row_points=True,
    #     row_points_size=10,
    #     show_row_labels=False,
    #     show_column_points=True,
    #     column_points_size=30,
    #     show_column_labels=False,
    #     legend_n_cols=1
    #  )

    # ax.get_figure()
    # plt.show()
    # %%
    from mpl_toolkits.mplot3d import Axes3D

    sns.set(style = "darkgrid")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    # compute the ratio of good vs bad experiments, by computing the mean of the flag for each unique value of the parameters
    # all_exps_df.groupby(['ping frequency', 'train load', 'moving average window', "flag"]).size().unstack()
    # all_exps_df = all_exps_df.fillna(0)
    # all_exps_df = all_exps_df.div(all_exps_df.sum(axis=1), axis=0) 

    x = all_exps_df[all_exps_df["flag"]==0.0]['ping frequency']
    y = all_exps_df[all_exps_df["flag"]==0.0]['train load']
    z = all_exps_df[all_exps_df["flag"]==0.0]["moving average window"]

    ax.set_xlabel('ping frequency')
    ax.set_ylabel('train load')
    ax.set_zlabel("moving average window")

    ax.scatter(x, y, z, label='bad')
    
    x = all_exps_df[all_exps_df["flag"]==1.0]['ping frequency']
    y = all_exps_df[all_exps_df["flag"]==1.0]['train load']
    z = all_exps_df[all_exps_df["flag"]==1.0]["moving average window"]
    ax.scatter(x, y, z, alpha=0.5, marker="*", label='good')

    plt.legend()
    plt.show()
    # %%
    import numpy as np
    import matplotlib.pyplot as plt

    x = all_exps_df["ping frequency"]
    y = all_exps_df["train load"]
    z = all_exps_df["moving average window"]
    w = all_exps_df["mse"]

    all_exps_df["ping frequency"].hist()
    

# %%
# for each of the variables : "ping frequency", "train load" and "moving average window", plot the proportion of having a "flag" = 1 or 0 in bar plots
def plot_bar(df, column):
    df = df[[column, "flag"]]
    df = df.groupby([column, "flag"]).size().unstack()
    df = df.fillna(0)
    df = df.div(df.sum(axis=1), axis=0)
    df.plot(kind="bar", stacked=True, figsize=(16,7))
plot_bar(all_exps_df, "ping frequency")
plot_bar(all_exps_df, "train load")
plot_bar(all_exps_df, "moving average window") 
plt.show()
# same as above but have in the x axis all the combination of the unique values of the 3 variables and sort them by the higher count of "flag" = 0, and invert the proportion of "flag" = 1 and 0 by having flag = 1 on the top, add also the legend where you put that 0 is refering to bad experiments in red and 1 to good experiments in green and add tight layout and xlabel and ylabel
# def plot_bar2(df, column1, column2, column3):
#     df = df[[column1, column2, column3, "flag"]]
#     df = df.groupby([column1, column2, column3, "flag"]).size().unstack()
#     df = df.fillna(0)
#     df = df.div(df.sum(axis=1), axis=0)
#     df = df.sort_values(by=[0.0], ascending=True)
#     df = df.reindex(columns=[1.0, 0.0])
#     df.plot(kind="bar", stacked=True, figsize=(16,7), color=["green", "#b02412"])
#     plt.legend(["good", "bad"], loc="upper left")
#     plt.tight_layout()
#     plt.xlabel("combination of the 3 variables : (ping frequency, train load, moving average window)")
#     plt.ylabel("proportion of good and bad experiments")
#     plt.show()

#%%
# use 3DContour plot method to visualize the distribution of "mse" given the 3 variables, the x, y and z axis should be the 3 variables and the color should be the "mse", add the label for the x, y and z axis and add the title "mse distribution given the 3 variables"
def plot_3d_contour(df, column1, column2, column3):
    df = df[[column1, column2, column3, "mse"]]
    df = df.groupby([column1, column2, column3]).mean().reset_index()
    df = df.pivot_table(index=column1, columns=column2, values="mse")
    df = df.sort_index(ascending=False)
    df = df.sort_index(axis=1, ascending=False)
    df = df.fillna(0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(df.columns, df.index, df.values, cmap='viridis')
    # add labels
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    ax.set_zlabel(column3)
    # add colorbar
    plt.tight_layout()
    plt.legend()
    
    ax.set_title("mse distribution given the 3 variables")
    plt.show()

# # same as above but use scatter plot instead of 3DContour plot

# %%
# compute fast fourier transformation of the "mse" column and plot the result in a bar plot, add the xlabel and ylabel and add the title "fast fourier transformation of the mse column"
