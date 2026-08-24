# Data Card: Machine Learning Education Preference Dataset

- **Dataset name**: Sample Preferences for Machine Learning Education (`sample_preferences.jsonl`)
- **Source**: Curated technical benchmark examples covering core machine learning, deep learning, and transformer architectures.
- **License/permission**: MIT License / Educational Use.
- **Schema**: JSON Lines format where each record is structured as:
  - `prompt` (`str`): Instructional question or concept prompt.
  - `chosen` (`str`): High-quality, technically accurate response.
  - `rejected` (`str`): Plausible but erroneous or oversimplified response.
  - `metadata` (`dict`): Annotation context including `domain` ("education") and `rubric` ("accuracy").
- **Labeling rubric**: High-quality chosen answers provide precise mathematical and conceptual descriptions. Rejected responses represent common student misconceptions or incorrect descriptions of algorithms.
- **Known biases**: Favors comprehensive explanatory responses over brief summaries; centered around ML theory and system design.
- **Safety/PII checks**: Verified free of personally identifiable information (PII), proprietary credentials, or safety violations.
- **Train/validation/test split method**: Prompt-grouped deterministic split (`seed=42`) preventing prompt overlap across splits.
