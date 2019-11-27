universe = vanilla
Initialdir = /scratch/cluster/msakthi/Desktop/maddy/cs388/NLP-finalproject/NLP-Neural-Networks-for-Sentiment-Analysis
Executable = /lusr/bin/bash
Arguments = /scratch/cluster/msakthi/Desktop/maddy/cs388/NLP-finalproject/NLP-Neural-Networks-for-Sentiment-Analysis/task.sh
+Group   = "GRAD"
+Project = "INSTRUCTIONAL"
+ProjectDescription = "NN Experiment"
Requirements = TARGET.GPUSlot
getenv = True
request_GPUs = 1
+GPUJob = true
Log = /scratch/cluster/msakthi/Desktop/maddy/cs388/NLP-finalproject/NLP-Neural-Networks-for-Sentiment-Analysis/condor/condor_28.log
Error = /scratch/cluster/msakthi/Desktop/maddy/cs388/NLP-finalproject/NLP-Neural-Networks-for-Sentiment-Analysis/condor/condor_28.err
Output = /scratch/cluster/msakthi/Desktop/maddy/cs388/NLP-finalproject/NLP-Neural-Networks-for-Sentiment-Analysis/condor/condor_28.out
Notification = complete
Notify_user = msakthi@cs.utexas.edu
Queue 1

