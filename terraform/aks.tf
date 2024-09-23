resource "azurerm_user_assigned_identity" "managed_identity_aks" {
  name                = "${var.component_name}-id"
  resource_group_name = azurerm_resource_group.resource_group.name
  location            = azurerm_resource_group.resource_group.location
}


resource "azurerm_kubernetes_cluster" "kubernetes_cluster" {
  for_each = var.aks_clusters

  name                = "aks-${each.key}"
  resource_group_name = azurerm_resource_group.resource_group.name
  location            = azurerm_resource_group.resource_group.location
  #dns_prefix_private_cluster          = var.component_name
  dns_prefix                          = random_string.pdns_prefix.id
  kubernetes_version                  = var.kubernetes_version
  node_resource_group                 = "${azurerm_resource_group.resource_group.name}_nodes"
  private_cluster_enabled             = false
  private_cluster_public_fqdn_enabled = false
  #private_dns_zone_id                 = azurerm_private_dns_zone.private_dns_zone.id

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.managed_identity_aks.id]
  }

  default_node_pool {
    name                         = each.value.default_node_pool_name
    vm_size                      = each.value.default_node_pool_vm_size
    os_disk_size_gb              = each.value.os_disk_size_gb
    os_disk_type                 = each.value.os_disk_type
    node_count                   = each.value.node_count
    only_critical_addons_enabled = true
    #vnet_subnet_id = azurerm_subnet.subnet.id
    node_labels = each.value.node_labels
    upgrade_settings {
      max_surge = 1
    }
    tags = each.value.tags
  }

  linux_profile {
    admin_username = var.admin_username

    ssh_key {
      key_data = var.ssh_public_key
    }
  }

  network_profile {
    network_plugin = each.value.network_plugin
    outbound_type  = each.value.outbound_type
  }

  tags = var.tags
}
