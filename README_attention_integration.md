# Attention Analysis Integration for job_util.py

This document explains how to use the updated `job_util.py` with integrated attention analysis and ATTRIEVAL functionality.

## Overview

The `job_util.py` has been enhanced to include attention visualization and ATTRIEVAL (attention-based retrieval) capabilities, similar to the MedCalc-Bench evaluation system. For every LLM prompt-response pair, the system can now:

1. **Extract attention weights** from the model's transformer layers
2. **Run ATTRIEVAL analysis** to identify top-k facts the model is paying attention to
3. **Save comprehensive analysis results** in the same directory structure as other logs

## Prerequisites

1. **attention_viz module**: Ensure the attention_viz module is available in your environment
2. **Local HuggingFace model**: The Qwen model (or any other local model) should be accessible
3. **Required dependencies**: torch, transformers, numpy, and attention_viz components

## Usage

### Basic Usage (without attention analysis)
```bash
python job_util.py medcalc_rules --lo 0 --hi 10
```

### With Attention Analysis Enabled
```bash
python job_util.py medcalc_rules --lo 0 --hi 10 --enable_attention_analysis
```

## Command Line Arguments

- `--enable_attention_analysis`: Enable attention analysis and ATTRIEVAL for each prompt-response pair
- All existing arguments from the original job_util.py are still supported

## Output Structure

When attention analysis is enabled, the system creates the following directory structure:

```
doctest-prompting-data/
└── logs2/
    └── [model_name]/
        └── [task]/
            ├── [task]_[lo]-[hi].log          # Standard log file
            ├── [task]_[lo]-[hi].json         # Standard JSON results
            └── top_k/                        # NEW: Attention analysis results
                ├── example_0000/
                │   ├── essential_attention_data.npz      # Compressed attention weights
                │   ├── attrieval_results.json            # Comprehensive retrieval results
                │   ├── attrieval_analysis_report.md      # Human-readable analysis
                │   ├── top_facts_summary.json            # Top retrieved facts
                │   ├── attrieval_attention_data.npz      # Attention weights and scores
                │   └── analysis_summary.json             # Metadata and file list
                ├── example_0001/
                │   └── ... (same structure)
                └── example_NNNN/
                    └── ... (same structure)
```

## Generated Files Per Example

For each prompt-response pair, the following files are generated:

### 1. `essential_attention_data.npz`
- **Format**: Compressed NumPy archive
- **Content**: Essential attention weights from key layers (first, middle, last)
- **Purpose**: Memory-efficient storage of attention patterns

### 2. `attrieval_results.json`
- **Format**: JSON
- **Content**: Comprehensive ATTRIEVAL analysis results
- **Purpose**: Complete fact retrieval data with scores and rankings

### 3. `attrieval_analysis_report.md`
- **Format**: Markdown
- **Content**: Human-readable analysis report
- **Purpose**: Easy-to-read summary of attention patterns and retrieved facts

### 4. `top_facts_summary.json`
- **Format**: JSON
- **Content**: Top retrieved facts with metadata
- **Purpose**: Quick access to the most important extracted information

### 5. `attrieval_attention_data.npz`
- **Format**: Compressed NumPy archive
- **Content**: Aggregated attention data and fact scores
- **Purpose**: Detailed attention analysis for research

### 6. `analysis_summary.json`
- **Format**: JSON
- **Content**: Metadata and file listing
- **Purpose**: Index of all generated files with timestamps

## Model Loading

The system automatically handles different model loading scenarios:

1. **HuggingFace Hub models**: Standard model identifiers (e.g., "Qwen/Qwen2.5-7B-Instruct")
2. **Local absolute paths**: Full paths to local model directories
3. **Local relative paths**: Checks `codebase/huggingface/` directory automatically

## ATTRIEVAL Configuration

The system uses the following default ATTRIEVAL configuration:
- **Layer fraction**: 0.25 (uses last 25% of model layers)
- **Top-k tokens**: 10 per chain-of-thought token
- **Frequency threshold**: 0.99 (filters attention sinks)
- **Max facts**: 10 retrieved facts per example

## Memory Management

The system includes automatic memory management:
- **Progressive cleanup**: Attention data is saved and cleared progressively
- **Memory-efficient storage**: Uses compressed NumPy formats
- **GPU cleanup**: Automatically clears CUDA cache when available
- **Model cleanup**: Removes attention models from memory after processing

## Example Workflow

1. **Setup**: Ensure attention_viz module and local models are available
2. **Run**: Execute job_util.py with `--enable_attention_analysis`
3. **Monitor**: Watch console output for attention analysis progress
4. **Review**: Check the `top_k/` directory for detailed attention analysis results
5. **Analyze**: Use the generated reports and data files for further research

## Error Handling

The system gracefully handles various error conditions:
- **Missing attention_viz**: Continues without attention analysis
- **Model loading failures**: Reports errors and continues with standard evaluation
- **Individual analysis failures**: Logs errors but continues processing other examples
- **Memory issues**: Implements cleanup and garbage collection

## Performance Considerations

- **Memory usage**: Attention analysis requires additional GPU/CPU memory
- **Processing time**: Adds 20-50% processing time per example
- **Storage**: Generates 5-20MB of additional data per example
- **Parallelization**: Currently processes examples sequentially for attention analysis

## Integration with Existing Workflows

The enhanced job_util.py maintains full backward compatibility:
- All existing functionality remains unchanged
- Attention analysis is purely additive
- Standard log and JSON outputs are unaffected
- Can be used as a drop-in replacement for the original job_util.py

## Troubleshooting

### Common Issues

1. **ImportError for attention_viz**:
   - Ensure attention_viz module is in Python path
   - Check if module exists in `codebase/attention_viz/`

2. **CUDA out of memory**:
   - Reduce batch size or example count
   - Use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`

3. **Model not found**:
   - Verify model path exists
   - Check `codebase/huggingface/` directory for local models

4. **Permission errors**:
   - Ensure write permissions for log directories
   - Check disk space availability

### Debug Mode

For debugging, run with a small example set:
```bash
python job_util.py medcalc_rules --lo 0 --hi 2 --enable_attention_analysis
```

This will process only 2 examples and allow you to verify the attention analysis setup.