location = "westeurope"

kubernetes_version = "1.28.9"

tags = {
  environment = "ephemeral"
  component   = "aks-prefect"
  region      = "eu-west"
  service     = "prefect"
}

component_name               = "aks-prefect"
backend_storage_account_name = "aksprefecttfstate"

admin_username = "amackojc"
ssh_public_key = ""

aks_clusters = {
  "prefect" = {
    default_node_pool_name    = "system"
    default_node_pool_vm_size = "Standard_B2ms"
    identity_type             = "SystemAssigned"
    node_count                = 1
    os_disk_size_gb           = 32
    os_disk_type              = "Managed"
    node_taints = [
      "CriticalAddonsOnly=true:NoSchedule"
    ]

    node_labels = {
      "nodepool-type" = "system"
      "environment"   = "prefect"
      "nodepool"      = "linux"
    }

    tags = {
      "nodepool-type" = "system"
      "environment"   = "prefect"
      "nodepoolos"    = "linux"
    }
  }
}

node_pools = {
  "prefect" = {
    vm_size             = "Standard_D2as_v5"
    cluster             = "prefect"
    enable_auto_scaling = true
    min_count           = 0
    max_count           = 3
    priority            = "Spot"
    eviction_policy     = "Delete"
    node_taints = [
      "kubernetes.azure.com/scalesetpriority=spot:NoSchedule",
      "sku=standard:NoSchedule"
    ]

    node_labels = {
      nodepool-type                           = "user"
      environment                             = "prefect"
      nodepool                                = "linux"
      "kubernetes.azure.com/scalesetpriority" = "spot"
    }

    tags = {
      nodepool-type = "user"
      environment   = "prefect"
      nodepoolos    = "linux"
    }
  }

  "gpuspot" = {
    vm_size             = "Standard_NC8as_T4_v3"
    cluster             = "prefect"
    enable_auto_scaling = true
    min_count           = 1
    max_count           = 3
    priority            = "Spot"
    eviction_policy     = "Delete"
    node_taints = [
      "kubernetes.azure.com/scalesetpriority=spot:NoSchedule",
      "sku=gpu:NoSchedule"
    ]

    node_labels = {
      nodepool-type                           = "gpu"
      environment                             = "prefect"
      nodepool                                = "linux"
      "kubernetes.azure.com/scalesetpriority" = "spot"
    }

    tags = {
      nodepool-type        = "user"
      environment          = "prefect"
      nodepoolos           = "linux"
      SkipGPUDriverInstall = true
    }
  }
}
