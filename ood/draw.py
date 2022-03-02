import seaborn as sns
import matplotlib.pyplot as plt


def draw_score_distr_plot(
    scores_distr, score_type, model_name, in_data_name, out_data_name
):

    sns.set_theme()
    plt.figure(figsize=(12, 8))
    sns.distplot(
        scores_distr[in_data_name],
        kde=True,
        norm_hist=True,
        hist=True,
        label=in_data_name,
    )
    sns.distplot(
        scores_distr[out_data_name],
        kde=True,
        norm_hist=True,
        hist=True,
        label=out_data_name,
    )

    plt.title(
        f"{model_name} Softmax score distribution for {in_data_name} and {out_data_name} datasets"
    )
    plt.xlabel(f"{score_type} score")

    plt.legend()
