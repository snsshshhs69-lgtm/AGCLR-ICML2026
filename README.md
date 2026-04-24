# AGCLR-ICML2026
Code for "Why Limit the Residual Stream to Layers and Not Tokens?"
# AGCLR: Adaptive Gated Continuous Latent Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Anonymous submission to ICML 2026**  
> Paper: *"Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning"*

---

## Overview

Large language models struggle with multi-step reasoning when intermediate states are lost across reasoning passes. We identify this **concept bottleneck** in vanilla CoCoNuT and propose **AGCLR** (Adaptive Gated Continuous Latent Reasoning), which augments CoCoNuT with a persistent gated memory stream.

**Key Results:**
- ✅ **GSM8K:** 34.0% accuracy (+2.6% over CoCoNuT)
- ✅ **HotpotQA:** 14.0% EM, 19.4% F1 (+3.6% EM over CoCoNuT)
- ✅ **ProsQA:** 96.0% accuracy (+4.0% over CoCoNuT)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM
- GPU with 24GB+ VRAM (recommended for full training)

### Quick Start

```bash
