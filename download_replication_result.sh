#!/bin/bash

rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/credit/evaluation-0_masked inprocess/evaluation-credit/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/credit/evaluation-0 inprocess/evaluation-credit/

rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/adult/evaluation-0_masked inprocess/evaluation-adult/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_distance/model/adult/evaluation-0 inprocess/evaluation-adult/




rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_postprocess/model/credit/evaluation-0_masked postprocess/evaluation-credit/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_postprocess/model/credit/evaluation-0 postprocess/evaluation-credit/

rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_postprocess/model/adult/evaluation-0_masked postprocess/evaluation-adult/
rsync -av --progress liutia25@hpcc.msu.edu:dev/fairness/mpvae/fair_through_postprocess/model/adult/evaluation-0 postprocess/evaluation-adult/