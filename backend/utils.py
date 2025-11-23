def normalize_description(df, inp_col_name='description', out_col_name='normalized_description'):
    df[out_col_name] = df[inp_col_name].str.lower().str.strip()
    return df

def create_explanation_column(df, merchant_col='extracted_merchant_name', context_col='context_feature', explanation_col='explanation'):
    def make_explanation(merchant, context):
        # Split merchant and context into words
        merchant_words = merchant.split() if isinstance(merchant, str) else []
        context_words = context.split() if isinstance(context, str) else []

        seen = set()
        unique_merchant_words = []
        unique_context_words = []

        # Add merchant words keeping only first occurrences
        for w in merchant_words:
            if w.lower() not in seen:
                unique_merchant_words.append(w)
                seen.add(w.lower())

        # Add context words ignoring words already in merchant
        for w in context_words:
            if w.lower() not in seen:
                unique_context_words.append(w)
                seen.add(w.lower())

        merchant_str = ' '.join(unique_merchant_words)
        context_str = ' '.join(unique_context_words)

        if merchant_str and context_str:
            return f'Categorization is due to "{merchant_str}" and "{context_str}"'
        elif merchant_str:
            return f'Categorization is due to "{merchant_str}"'
        elif context_str:
            return f'Categorization is due to "{context_str}"'
        else:
            return 'Categorization details unavailable'

    df[explanation_col] = df.apply(lambda row: make_explanation(row[merchant_col], row[context_col]), axis=1)

    return df
