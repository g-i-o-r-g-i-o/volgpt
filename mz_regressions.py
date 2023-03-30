import re
import pandas as pd
import statsmodels.api as sm

def mz_regressions(text):

    def extract_sequences(text):
        rr_seq = [len(x) for x in re.findall(r'rr+', text)]
        lr_seq = [len(x) for x in re.findall(r'll+', text)]
        return rr_seq, lr_seq

    generated_text = train_and_generate(text_file_path, max_new_tokens=5000)
    test_text = decode(test_data.tolist())

    gen_rr_seq, gen_lr_seq = extract_sequences(generated_text)
    test_rr_seq, test_lr_seq = extract_sequences(test_text)

    def prepare_data_for_regression(gen_seq, test_seq):
        assert len(gen_seq) == len(test_seq), "Generated and test sequences must have the same length."
        data = pd.DataFrame({'gen_seq': gen_seq, 'test_seq': test_seq})
        return data

    rr_data = prepare_data_for_regression(gen_rr_seq, test_rr_seq)
    lr_data = prepare_data_for_regression(gen_lr_seq, test_lr_seq)

    def perform_mz_regression(data):
        X = data['gen_seq']
        X = sm.add_constant(X)  # Add a constant to fit an intercept
        y = data['test_seq']
        model = sm.OLS(y, X).fit()
        return model

    rr_model = perform_mz_regression(rr_data)
    lr_model = perform_mz_regression(lr_data)

    return rr_model, lr_model
