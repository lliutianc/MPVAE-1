#!/bin/bash

rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/credit/evaluation-0_masked evaluation-credit/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/credit/evaluation-0 evaluation-credit/

rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/adult/evaluation-0_masked evaluation-adult/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/adult/evaluation-0 evaluation-adult/