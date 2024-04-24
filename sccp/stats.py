import numpy as np
import pandas as pd
import statsmodels.api as sm


def wls_stats_comparison(df, column_comparison_name, category_name, status_name):
    """Calculates whether cells are statistically signicantly different"""
    pval_df = pd.DataFrame()
    df["Y"] = 1
    df.loc[df[category_name] == status_name, "Y"] = 0
    for cell in df["Cell Type"].unique():
        Y = df.loc[df["Cell Type"] == cell][column_comparison_name].to_numpy()
        X = df.loc[df["Cell Type"] == cell]["Y"].to_numpy()
        weights = np.power(df.loc[df["Cell Type"] == cell]["Cell Count"].values, 1)
        mod_wls = sm.WLS(Y, sm.tools.tools.add_constant(X), weights=weights)
        res_wls = mod_wls.fit()
        pval_df = pd.concat(
            [
                pval_df,
                pd.DataFrame(
                    {
                        "Cell Type": [cell],
                        "p Value": res_wls.pvalues[1]
                        * df["Cell Type"].unique().size,
                    }
                ),
            ]
        )

    return pval_df
