"""
@author: daviddeepwell
"""

from patsy import dmatrices
import statsmodels.formula.api as smf
import statsmodels.api as sm

def make_model_fit(train_data, test_data, quant_list):
    
    # create the equation to fit
    all_vars = " + ".join(quant_list)
    expr = """Counts ~ weekofyear + """ + all_vars
    
    y_train, x_train = dmatrices(expr, train_data, return_type='dataframe')
    y_test,  x_test  = dmatrices(expr, test_data,  return_type='dataframe')

    # Run Poisson model to find alpha using auxiliary OLS regression
    poisson_training_results = sm.GLM(y_train, x_train, family=sm.families.Poisson()).fit()
    train_data['LAMBDA'] = poisson_training_results.mu
    
    train_data['aux_var'] = train_data.apply(\
        lambda x: ((x['Counts'] - x['LAMBDA'])**2 - x['LAMBDA']) / x['LAMBDA'], axis=1)
        
    ols_expr = """aux_var ~ LAMBDA - 1"""
    aux_olsr_results = smf.ols(ols_expr, train_data).fit()
    alph = aux_olsr_results.params[0]
    
    print('alpha: ',alph)
    print(aux_olsr_results.params)
    print('t values:')
    print(aux_olsr_results.tvalues)
    
    # run NG model with alpha value
    nb2_training_results = sm.GLM(\
        y_train, x_train,family=sm.families.NegativeBinomial(alpha=alph)).fit()

    # make prediction
    nb2_predictions = nb2_training_results.get_prediction(x_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    
    # get the training fit
    nb2_predictions_train = nb2_training_results.get_prediction(x_train)
    predictions_summary_frame_train = nb2_predictions_train.summary_frame()    

    return predictions_summary_frame, y_test, x_test, predictions_summary_frame_train