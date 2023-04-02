# statistical analysis of volgpt performance
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from volgpt_clean_data import clean_data

def volgpt_stats(generated_text, test_data, itos):

    def tensor_to_string(tensor, itos):
        return ''.join([itos[i] for i in tensor.tolist()])

    test_data_text = test_data
    column_names = ['DateTimeIndex', 'Ticker', 'CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'rr', 'lr']

    # use the itos mapping with the tensor_to_string function to convert the tensors to strings
    test_data_text = tensor_to_string(test_data, itos)

    # use the clean_data function in clean_data.py to clean the preds and the test_data
    _, generated_clean, _ = clean_data(generated_text, column_names)
    _, test_data_clean, _ = clean_data(test_data_text, column_names)

    # ensure the timing of the predictions and the test data are aligned by merging on DateTimeIndex
    merged_data = generated_clean.merge(test_data_clean, on='DateTimeIndex', suffixes=('_generated', '_test'))

    # remove rows with spurious values for rr_generated and/or lr_generated 
    merged_data = merged_data[(merged_data['rr_generated'] <= 2) & (merged_data['rr_generated'] >= -2) & 
                                (merged_data['lr_generated'] <= 1) & (merged_data['lr_generated'] >= -1)]
    
    # calculate MSE and MAE for the raw returns (rr) and log returns (lr)
    rr_mse = mean_squared_error(merged_data['rr_generated'], merged_data['rr_test'])
    rr_mae = mean_absolute_error(merged_data['rr_generated'], merged_data['rr_test'])

    lr_mse = mean_squared_error(merged_data['lr_generated'], merged_data['lr_test'])
    lr_mae = mean_absolute_error(merged_data['lr_generated'], merged_data['lr_test'])

    # Get the true and predicted raw returns and log returns
    true_raw_returns = merged_data['rr_test']
    predicted_raw_returns = merged_data['rr_generated']

    true_log_returns = merged_data['lr_test']
    predicted_log_returns = merged_data['lr_generated']

    # Perform paired t-tests
    raw_t_stat, raw_p_value = stats.ttest_rel(true_raw_returns, predicted_raw_returns)
    log_t_stat, log_p_value = stats.ttest_rel(true_log_returns, predicted_log_returns)

    # Print the results
    print("Clean generated data: ")
    print(generated_clean), print()

    print("Clean test data: ")
    print(test_data_clean), print()

    print("Merged data: ")
    print(merged_data), print()

    print("Generated data date range: ", generated_clean['DateTimeIndex'].min(), "to", generated_clean['DateTimeIndex'].max())
    print("Test data date range: ", test_data_clean['DateTimeIndex'].min(), "to", test_data_clean['DateTimeIndex'].max()), print()

    print(f"Raw returns MSE: {rr_mse:.4f}, MAE: {rr_mae:.4f}")
    print(f"Log returns MSE: {lr_mse:.4f}, MAE: {lr_mae:.4f}"), print()

    print(f"Raw returns paired t-test results: T-statistic = {raw_t_stat:.2f}, p-value = {raw_p_value:.6f}")
    print(f"Log returns paired t-test results: T-statistic = {log_t_stat:.2f}, p-value = {log_p_value:.6f}")

    return generated_clean, test_data_clean, merged_data, rr_mae, rr_mse, lr_mae, lr_mse, raw_t_stat, raw_p_value, log_t_stat, log_p_value
