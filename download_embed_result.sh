#!/bin/bash

# mkdir embed-0/evaluation-credit embed-0/evaluation-adult

rsync -av --progress liu3351@dlm.ecn.purdue.edu:~/dev/mpvae/fair_through_distance/model/credit/probability probability-0/evaluation-credit/
rsync -av --progress liu3351@dlm.ecn.purdue.edu:~/dev/mpvae/fair_through_distance/model/credit/probability probability-0/evaluation-credit/

# rsync -av --progress liu3351@dlm.ecn.purdue.edu:~/dev/mpvae/fair_through_distance/model/credit/embedding embed-0/evaluation-credit/
# rsync -av --progress liu3351@dlm.ecn.purdue.edu:~/dev/mpvae/fair_through_distance/model/credit/embedding embed-0/evaluation-credit/
