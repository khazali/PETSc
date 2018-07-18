#!/usr/bin/env groovy

pipeline {
    agent any
    triggers{
        bitbucketpr(projectPath:'<BIT_BUCKET_PATH>',
        cron:'H/15 * * * *',
        credentialsId:'',
        username:'',
        password:'',
        repositoryOwner:'',
        repositoryName:'',
        branchesFilter:'',
        branchesFilterBySCMIncludes:false,
        ciKey:'',
        ciName:'',
        ciSkipPhrases:'',
        checkDestinationCommit:false,
        approveIfSuccess:false,
        cancelOutdatedJobs:true,
        commentTrigger:'')
    }
}
