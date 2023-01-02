# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# Generate a local
julia --project=@. -e 'using Pkg; Pkg.add("LocalCoverage"); using LocalCoverage; cov = generate_coverage("BayesNet"; run_test=true); html_coverage(cov; dir="coverage_html");'
