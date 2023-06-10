Don't use 2.0 defintion

In deployment.yml, need list of environments:

```yaml
custom:

  # Cluster configs for each environment
  default-cluster-spec: &default-cluster-spec
    spark_version: '11.0.x-cpu-ml-scala2.12'
    node_type_id: 'Standard_DS3_v2'
    driver_node_type_id: 'Standard_DS3_v2'
    num_workers: 1
    # To reduce start up time for each job, it is advisable to use a cluster pool. To do so involves supplying the following
    # two fields with a pool_id to acquire both the driver and instances from.
    # If driver_instance_pool_id and instance_pool_id are set, both node_type_id and driver_node_type_id CANNOT be supplied.
    # As such, if providing a pool_id for driver and worker instances, please ensure that node_type_id and driver_node_type_id are not present
#    driver_instance_pool_id: '0617-151415-bells2-pool-hh7h6tjm'
#    instance_pool_id: '0617-151415-bells2-pool-hh7h6tjm'

  dev-cluster-config: &dev-cluster-config
    new_cluster:
      <<: *default-cluster-spec

  staging-cluster-config: &staging-cluster-config
    new_cluster:
      <<: *default-cluster-spec

  prod-cluster-config: &prod-cluster-config
    new_cluster:
      <<: *default-cluster-spec

environments:
    default:
        workflows:
            - name: "name-of-worflow"
              job_clusters:

            tasks:
                - task_key: "main"
                <<: *basic-static-cluster
                python_wheel_task:
                    package_name:
                    entry_point:
                    parameters:

```
Environments must correlate with environments in `projects.json` to point where deployment should go 

What are task_keys/how to define?

job_cluster_keys reference the job_clusters

Can add dependencies between tasks, by task keys:
```yaml
tasks:
  - task_key: "feature-creation"
    job_cluster: "default"
    spark_python_task:
      python_file: "path-to-file"
      parameters: ["--conf-file", "path/to/conf.yaml"] # this file is
  - task_key: "train-model"
    depends_on: 
      - task_key: "feature-creation"
    job_cluster_key: "default" 
    python_wheel_task:
     ...

```

# How to deploy via VC repo
`dbx deploy --assets-only` 