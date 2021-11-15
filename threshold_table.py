# Dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import pandas_flavor as pf

@pf.register_dataframe_method
def threshold_table(model, 
                    X_train,
                    y_train,
                    tn = 1, 
                    fp = 1,
                    fn = 1,
                    tp = 1,
                    top_n = 100,
                    total_threshold = 100,
                    positive_values_color = "#5fba7d",
                    negative_values_color = '#d65f5f',
                    column_label_position = 'center',
                    cell_label_position   = 'center',
                    output_type = "pandas_style"
                    ):
    """
    Generate a value-based, ranked-ordered 'Threshold Table' for the True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN) counts, normalized
    percentages, and values at classification model thresholds.
    
    Args: 
      model ([model object]): 
        Model object generated from sklearn or xgboost, e.g., xgboost.sklearn.XGBClassifier.
        
      X_train ([pandas.DataFrame]): 
        X_train data.
        
      y_train ([pandas.Series]): 
        y_train data.
        
      tn ([float, optional)]: 
        Float specifying the monetary value (weight) of a true negative. Default weight is 1.
      
      fp ([float, optional)]: 
        Float specifying the monetary value (weight) of a false positive. Default weight is 1.
      
      fn ([float, optional)]: 
        Float specifying the monetary value (weight) of a false negative. Default weight is 1.
      
      tp ([float, optional)]: 
        Float specifying the monetary value (weight) of a true positive. Default weight is 1.
      
      top_n ([float, optional)]: 
        Float specifying the top N ranked thresholds by monetary value of model. Default is 100.
      
      total_threshold ([float, optional)]: 
        Float specifying the total number of thresholds to test, e.g., 10, 100, 1000. Default is 100.
      
      positive_values_color ([str, optional]): 
        String specifying the color of the positively valued amounts. Defaults to "#5fba7d".
      
      negative_values_color ([str, optional]):
        String specifying the color of the positively valued amounts. Defaults to "#d65f5f".
      
      column_label_position ([str, optional]):
        String specifying the position of the column labels. Defaults to "center".
      
      cell_label_position ([str, optional]): 
        String specifying the position of the cell values. Defaults to "center".
      
      output_type ([str, optional]): 
        String of either 'pandas_style' or 'pandas_dataframe' specifying if a styled table of the counts, percentages,
        and values of TP, TN, FP, and FN are supplied as a pandas style object or a pandas dataframe. Default is 'pandas_style'.
           
    Returns:
        if output_type=="pandas_style":
        [pandas.io.formats.style.Styler] of TP, TN, FP, and FN values at each decile threshold.
        
        if output_type=="pandas_dataframe":
        [pandas.DataFrame] of TP, TN, FP, and FN values in the form of a confusion plot.  

    Example Tables:
    
      >>> from sklearn.linear_model import LogisticRegression
      >>> from sklearn.model_selection import train_test_split
      >>> from sklearn.datasets import make_classification
      >>> X, y = make_classification(n_samples=50000, n_features=2, n_redundant=0,
      >>> n_clusters_per_class=2, weights=[0.50], flip_y=0, random_state=123)
      >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
      >>> model = LogisticRegression(solver='lbfgs')
      >>> model.fit(X_train, y_train)
      >>> # Top 100 Pandas Style
      >>> top100  = black_red_green_table(model, X_train, y_train, tn=10.75, fp=-32.03, fn=-150.87, tp=80.14, top_n=100)
      >>> # Top 10 Pandas Stype
      >>> top10   = threshold_table(model, X_train, y_train, tn=10.75, fp=-32.03, fn=-150.87, tp=80.14, top_n=10, total_threshold = 10)
      >>> # Top 1000 Pandas Style
      >>> top1000 = threshold_table(model, X_train, y_train, tn=10.75, fp=-32.03, fn=-150.87, tp=80.14, top_n=1000, total_threshold = 1000)
      >>> # Top 100 Pandas DataFrame
      >>> pandas100 = threshold_value_table(model, X_train, y_train, tn=10.75, fp=-32.03, fn=-150.87, tp=80.14, top_n=100, output_type='pandas_dataframe')
      >>> # Export pandas100 to Excel
      >>> top100.to_excel('styled.xlsx', engine='openpyxl')
    """

    def get_df(X_t,y_t,t):
    
        # Baseline Metrics
        threshold = 0.0
    
        # Classify y_pred based upon thresholds
        y_pred = (model.predict_proba(X_t)[:, 1] > t).astype('float')
    
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
    
        # Threshold DataFrame
        return pd.DataFrame([tn, fp, fn, tp]) \
        .rename(index = {0:'TN Count', 1:'FP Count', 2:'FN Count', 3:'TP Count'}) \
        .rename(columns = {0:t})

    # Concat All Threshold Dataframes
    threshold_table = pd.concat([get_df(X_train,y_train,i/total_threshold) for i in range(0,total_threshold+1)],axis=1)
   
    # Transpose DataFrame
    threshold_table = threshold_table.T
    
    # True Negative Normalized Percentages
    threshold_table['TN %'] = ((threshold_table['TN Count']) / (threshold_table['TN Count'] + threshold_table['FP Count'] + threshold_table['FN Count'] + threshold_table['TP Count'])) * 100
    
    # False Negative Normalized Percentages
    threshold_table['FN %'] = ((threshold_table['FN Count']) / (threshold_table['TN Count'] + threshold_table['FP Count'] + threshold_table['FN Count'] + threshold_table['TP Count'])) * 100
    
    # False Positive Normalized Percentages
    threshold_table['FP %'] = ((threshold_table['FP Count']) / (threshold_table['TN Count'] + threshold_table['FP Count'] + threshold_table['FN Count'] + threshold_table['TP Count'])) * 100
    
    # True Positive Normalized Percentages
    threshold_table['TP %'] = ((threshold_table['TP Count']) / (threshold_table['TN Count'] + threshold_table['FP Count'] + threshold_table['FN Count'] + threshold_table['TP Count'])) * 100
    
    # True Negative Monetary Value
    threshold_table['TN Value'] = threshold_table['TN Count'] * tn
    
    # False Positive Monetary Value
    threshold_table['FP Value'] = threshold_table['FP Count'] * fp
    
    # False Negative Monetary Value
    threshold_table['FN Value'] = threshold_table['FN Count'] * fn
    
    # True Positive Monetary Value
    threshold_table['TP Value'] = threshold_table['TP Count'] * tp
    
    # Model Value
    threshold_table['Model Value'] = threshold_table['TN Value'] + threshold_table['FP Value'] + threshold_table['FN Value'] + threshold_table['TP Value']
    
    # Create Model Threshold Ranking
    threshold_table['Rank'] = threshold_table['Model Value'].rank(ascending=False)
    
    # Sort by Model Rank
    final = threshold_table.sort_values(by='Rank')
    
    # Change Model Rank to Integer
    final['Rank'] = final['Rank'].astype(int)
    
    # Make Threshold Index
    final.index.names = ['Threshold']
    
    # Reset Index
    final = final.reset_index()
    
    # Change Column Order
    final = final[['Rank', 'Threshold', 'TN Count','TN %', 'TN Value', 'FN Count', 'FN %', 'FN Value', 'FP Count', 'FP %', 'FP Value', 'TP Count', 'TP %','TP Value','Model Value']]
    
    # Outputs Threshold Table as Pandas Dataframe
    if output_type=="pandas_dataframe":

         return final
    
    # Outputs Threshold Table as Pandas Style Object
    if output_type=="pandas_style":
    
        # Color Mapping for Table 
        cvals  = [(final[['TN Value', 'FN Value', 'FP Value', 'TP Value']].min().min()), 0, (final[['TN Value', 'FN Value', 'FP Value', 'TP Value']].max().max())]
        colors = [negative_values_color, "white", positive_values_color]
        norm   = plt.Normalize(min(cvals), max(cvals))
        tuples = list(zip(map(norm,cvals), colors))

        cmap   = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        # Style Table
        final_style = final.head(top_n).style.format({"Model Value": "${:20,.2f}", 
                                                      "TN Value":"${:20,.2f}", 
                                                      "FP Value":"${:20,.2f}",
                                                      "FN Value":"${:20,.2f}",
                                                      "TP Value":"${:20,.2f}",
                                                      "TN Count":"{:20,.0f}",
                                                      "FP Count":"{:20,.0f}",
                                                      "FN Count":"{:20,.0f}",
                                                      "TP Count":"{:20,.0f}",
                                                      "TN %":"{0:.2f}%",
                                                      "FP %":"{0:.2f}%",
                                                      "FN %":"{0:.2f}%",
                                                      "TP %":"{0:.2f}%",
                                                      "Threshold":"{:20,.3f}"}) \
                    .bar(subset=["Model Value"], align='zero', color=[negative_values_color, positive_values_color]) \
                    .background_gradient(subset=["TN Value", "FP Value", "FN Value", "TP Value"], cmap=cmap, axis=None) \
                    .set_table_styles([dict(selector='th', props=[('text-align', column_label_position)])]) \
                    .set_properties(**{'text-align': cell_label_position}) \
                    .hide_index()

        return final_style
