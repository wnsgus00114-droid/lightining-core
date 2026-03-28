import Exia as ex


def main():
    pipe = ex.load_hf_pipeline(
        "sentiment-analysis",
        "distilbert-base-uncased-finetuned-sst-2-english",
    )
    print(pipe("Exia makes Lightning Core easier to use."))


if __name__ == "__main__":
    main()
