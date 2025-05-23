import json

# Define the updated notebook content
notebook = {
  "cells": [
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import json\n",
        "import pandas as pd\n",
        "from deepeval.test_case import LLMTestCase\n",
        "from deepeval.metrics import GEval\n",
        "from deepeval import assert_test\n",
        "\n",
        "# Load model outputs — each element is a JSON string\n",
        "with open(\"/Users/zhangran/Desktop/BP@UnitedStates/Code/D2D_Data2Dashboard/notebooks/exp01_domain_stim_harnow_output.json\", \"r\") as f:\n",
        "    output = json.load(f)\n",
        "\n",
        "insight = (\n",
        "    output[\"analysis\"][\"descriptive\"] + \" \" +\n",
        "    output[\"analysis\"][\"predictive\"] + \" \" +\n",
        "    output[\"analysis\"][\"domain_related\"]\n",
        ")\n",
        "\n",
        "# 3. Create a test case without reference output\n",
        "test_case = LLMTestCase(\n",
        "    input=\"Customer Relationship Management dataset with features like acquisition channel, retention, churn, periods active, etc.\",\n",
        "    actual_output=insight,\n",
        ")\n",
        "\n",
        "# 4. Define GEval metrics (self-evaluation — no expected_output)\n",
        "insightful = GEval(\n",
        "    name=\"Insightful\",\n",
        "    criteria=\"Does the output offer a deep or non-obvious understanding? Does it connect patterns or trends that aren't immediately apparent?\",\n",
        "    evaluation_params=[\"input\", \"actual_output\"],\n",
        ")\n",
        "\n",
        "novelty = GEval(\n",
        "    name=\"Novelty\",\n",
        "    criteria=\"Does the output go beyond generic interpretation? Would it surprise or teach something new to a domain expert?\",\n",
        "    evaluation_params=[\"input\", \"actual_output\"],\n",
        ")\n",
        "\n",
        "domain_relevance = GEval(\n",
        "    name=\"Domain Relevance\",\n",
        "    criteria=\"Is the output specific to the CRM domain? Does it reference domain-specific terms or relationships?\",\n",
        "    evaluation_params=[\"input\", \"actual_output\"],\n",
        ")\n",
        "\n",
        "# 5. Run evaluation (assertion-based print)\n",
        "print(\"\\n=== Insight Evaluation Report ===\")\n",
        "assert_test(test_case, [insightful, novelty, domain_relevance])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.6"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

# Write the updated notebook
with open('notebooks/exp01_domain_stim_harnow_eval.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2) 