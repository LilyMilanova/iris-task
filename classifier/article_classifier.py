import json
from pathlib import Path
from typing import List

import nltk
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from classifier.tools.common import MAX_LENGTH, preprocess_abstract

_classifier_instance = None


def get_classifier(sample_size=87):
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = BertBasedArXivClassifier(sample_size=sample_size)
    return _classifier_instance


class BertBasedArXivClassifier:
    def __init__(
        self,
        model_name: str = "bert-tiny",
        model_path: Path = Path(__file__).parent / "tools",
        sample_size: int = 1,
    ):
        """
        Initializes the classifier by loading the tokenizer, model, and category mappings.
        Args:
            model_name (str): Name of the model directory.
            model_path (Path): Base path where the model directory is located.
            sample_size (int): Sample size used in the cat2id filename.
        """
        self.model_dir = model_path / f"{model_name}{sample_size}"
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory '{self.model_dir}' does not exist.")

        self.cat2id_path = model_path / f"cat2id-sample_{sample_size}%.json"
        if not self.cat2id_path.exists():
            raise FileNotFoundError(f"Category to ID mapping file not found in '{self.model_dir}'.")

        with open(self.cat2id_path, "r") as f:
            self.cat2id = json.load(f)
        self.id2cat = {int(v): k for k, v in self.cat2id.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, num_labels=len(self.id2cat), problem_type="multi_label_classification",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "mps" if torch.backends.mps.is_available() else "cpu")

        nltk.download("stopwords")
        nltk.download("wordnet")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, abstract: str, threshold: float = 0.50) -> List[str]:
        """
        Predicts the categories for a given abstract.
        """
        abstract = preprocess_abstract(abstract).replace('\n', '')
        inputs = self.tokenizer(
            abstract,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Perform inference without gradient calculation
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # Apply sigmoid for multi-label probabilities
            preds = (probs > threshold).int().cpu().numpy()[0]  # Binary predictions

        return [self.id2cat[i] for i, pred in enumerate(preds) if pred == 1]


if __name__ == "__main__":
    test_entries = [
        {
            "abstract": "  In this paper, we show that a minimally coupled 3-form endowed with a proper\npotential can support a regular black hole interior. By choosing an appropriate\nform for the metric function representing the radius of the 2-sphere, we solve\nfor the 3-form field and its potential. Using the obtained solution, we\nconstruct an interior black hole spacetime which is everywhere regular. The\nsingularity is replaced with a Nariai-type spacetime, whose topology is\n$\\text{dS}_2 \\times \\text{S}^2$, in which the radius of the 2-sphere is\nconstant. So long as the interior continues to expand indefinitely, the\ngeometry becomes essentially compactified. The 2-dimensional de Sitter geometry\nappears despite the negative potential of the 3-form field. Such a dynamical\ncompactification could shed some light on the origin of de Sitter geometry of\nour Universe, exacerbated by the Swampland conjecture. In addition, we show\nthat the spacetime is geodesically complete. The geometry is singularity-free\ndue to the violation of the null energy condition.\n",
            "categories": "gr-qc hep-th"
        },
        {
            "abstract": "  This paper deals with the weak error estimates of the exponential Euler\nmethod for semi-linear stochastic partial differential equations (SPDEs). A\nweak error representation formula is first derived for the exponential\nintegrator scheme in the context of truncated SPDEs. The obtained formula that\nenjoys the absence of the irregular term involved with the unbounded operator\nis then applied to a parabolic SPDE. Under certain mild assumptions on the\nnonlinearity, we treat a full discretization based on the spectral Galerkin\nspatial approximation and provide an easy weak error analysis, which does not\nrely on Malliavin calculus.\n",
            "categories": "math.NA math.PR",
        },
        {
            "abstract": "  The processes of energy gain and redistribution in a dense gas subject to an\nintense ultrashort laser pulse are investigated theoretically for the case of\nhigh-pressure argon. The electrons released via strong-field ionization and\ndriven by oscillating laser field collide with neutral neighbor atoms, thus\neffecting the energy gain in the emerging electron gas via a short-range\ninverse Bremsstrahlung interaction. These collisions also cause excitation and\nimpact ionization of the atoms thus reducing the electron-gas energy. A kinetic\nmodel of these competing processes is developed which predicts the prevalence\nof excited atoms over ionized atoms by the end of the laser pulse. The creation\nof a significant number of excited atoms during the pulse in high-pressure\ngases is consistent with the delayed ionization dynamics in the pulse wake,\nrecently discovered by Gao et al.[1] This energy redistribution mechanism\noffers an approach to manage effectively the excitation vs. ionization patterns\nin dense gases interacting with intense laser pulses and thus opens new avenues\nfor diagnostics and control in these settings.\n",
            "categories": "physics.optics physics.plasm-ph",
        },
        {
            "abstract": "  We present a systematic discretization scheme for the Kardar-Parisi-Zhang\n(KPZ) equation, which correctly captures the strong-coupling properties of the\ncontinuum model. In particular we show that the scheme contains no finite-time\nsingularities in contrast to conventional schemes. The implications of these\nresults to i) previous numerical integration of the KPZ equation, and ii) the\nnon-trivial diversity of universality classes for discrete models of `KPZ-type'\nare examined. The new scheme makes the strong-coupling physics of the KPZ\nequation more transparent than the original continuum version and allows the\npossibility of building new continuum models which may be easier to analyse in\nthe strong-coupling regime.\n",
            "categories": "cond-mat",
        },
        {
            "abstract": "  For a slim, planar, semimodular lattice, G. Cz\\'edli and E.\\,T. Schmidt\nintroduced the fork extension in 2012. In this note we prove that the fork\nextension has the Congruence Extension Property. This paper has been merged\nwith Part II, under the title Congruences of fork extensions of slim\nsemimodular lattices, see arXiv: 1307.8404\n",
            "categories": "math.RA",
        },
        {
            "abstract": "  Black branes in AdS5 appear in a four parameter family labeled by their\nvelocity and temperature. Promoting these parameters to Goldstone modes or\ncollective coordinate fields -- arbitrary functions of the coordinates on the\nboundary of AdS5 -- we use Einstein's equations together with regularity\nrequirements and boundary conditions to determine their dynamics. The resultant\nequations turn out to be those of boundary fluid dynamics, with specific values\nfor fluid parameters. Our analysis is perturbative in the boundary derivative\nexpansion but is valid for arbitrary amplitudes. Our work may be regarded as a\nderivation of the nonlinear equations of boundary fluid dynamics from gravity.\nAs a concrete application we find an explicit expression for the expansion of\nthis fluid stress tensor including terms up to second order in the derivative\nexpansion.\n",
            "categories": "hep-th gr-qc nucl-th",
        },
        {
            "abstract": "  In this paper, we propose Push-SAGA, a decentralized stochastic first-order\nmethod for finite-sum minimization over a directed network of nodes. Push-SAGA\ncombines node-level variance reduction to remove the uncertainty caused by\nstochastic gradients, network-level gradient tracking to address the\ndistributed nature of the data, and push-sum consensus to tackle the challenge\nof directed communication links. We show that Push-SAGA achieves linear\nconvergence to the exact solution for smooth and strongly convex problems and\nis thus the first linearly-convergent stochastic algorithm over arbitrary\nstrongly connected directed graphs. We also characterize the regimes in which\nPush-SAGA achieves a linear speed-up compared to its centralized counterpart\nand achieves a network-independent convergence rate. We illustrate the behavior\nand convergence properties of Push-SAGA with the help of numerical experiments\non strongly convex and non-convex problems.\n",
            "categories": "cs.LG cs.DC cs.MA cs.SY eess.SY stat.ML",
        },
        {
            "abstract": "  The goal of a Question Paraphrase Retrieval (QPR) system is to retrieve\nequivalent questions that result in the same answer as the original question.\nSuch a system can be used to understand and answer rare and noisy\nreformulations of common questions by mapping them to a set of canonical forms.\nThis has large-scale applications for community Question Answering (cQA) and\nopen-domain spoken language question answering systems. In this paper we\ndescribe a new QPR system implemented as a Neural Information Retrieval (NIR)\nsystem consisting of a neural network sentence encoder and an approximate\nk-Nearest Neighbour index for efficient vector retrieval. We also describe our\nmechanism to generate an annotated dataset for question paraphrase retrieval\nexperiments automatically from question-answer logs via distant supervision. We\nshow that the standard loss function in NIR, triplet loss, does not perform\nwell with noisy labels. We propose smoothed deep metric loss (SDML) and with\nour experiments on two QPR datasets we show that it significantly outperforms\ntriplet loss in the noisy label setting.\n",
            "categories": "cs.CL cs.AI cs.IR cs.LG",
        },
        {
            "abstract": "  The hadronic decays of the tau lepton can be used to determine the effective\ncharge alpha_tau(m^2_tau') for a hypothetical tau-lepton with mass in the range\n0 < m_tau' < m_tau. This definition provides a fundamental definition of the\nQCD coupling at low mass scales. We study the behavior of alpha_tau at low mass\nscales directly from first principles and without any renormalization-scheme\ndependence by looking at the experimental data from the OPAL Collaboration. The\nresults are consistent with the freezing of the physical coupling at mass\nscales s = m^2_tau' of order 1 GeV^2 with a magnitude alpha_tau ~ 0.9 +/- 0.1.\n",
            "categories": "hep-ph hep-ex hep-th",
        },
        {
            "abstract": "  The sparse LiDAR point clouds become more and more popular in various\napplications, e.g., the autonomous driving. However, for this type of data,\nthere exists much under-explored space in the corresponding compression\nframework proposed by MPEG, i.e., geometry-based point cloud compression\n(G-PCC). In G-PCC, only the distance-based similarity is considered in the\nintra prediction for the attribute compression. In this paper, we propose a\nnormal-based intra prediction scheme, which provides a more efficient lossless\nattribute compression by introducing the normals of point clouds. The angle\nbetween normals is used to further explore accurate local similarity, which\noptimizes the selection of predictors. We implement our method into the G-PCC\nreference software. Experimental results over LiDAR acquired datasets\ndemonstrate that our proposed method is able to deliver better compression\nperformance than the G-PCC anchor, with $2.1\\%$ gains on average for lossless\nattribute coding.\n",
            "categories": "eess.IV",
        },
    ]

    classifier = get_classifier(sample_size=87)

    for e in test_entries:
        categories = sorted(classifier.predict(e["abstract"]))
        expected = sorted(e["categories"].split())
        matched = sorted(set(categories).intersection(expected))
        print(f'Matched:\t{" | ".join(matched)}\nPredicted:\t{" | ".join(categories)}\nExpected:\t{" | ".join(expected)}\n')
