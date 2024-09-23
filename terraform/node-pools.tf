resource "azurerm_kubernetes_cluster_node_pool" "kubernetes_cluster_nodepool" {
  for_each = var.node_pools

  name                  = each.key
  enable_auto_scaling   = each.value.enable_auto_scaling
  max_count             = each.value.max_count
  min_count             = each.value.min_count
  kubernetes_cluster_id = azurerm_kubernetes_cluster.kubernetes_cluster[each.value.cluster].id
  orchestrator_version  = azurerm_kubernetes_cluster.kubernetes_cluster[each.value.cluster].kubernetes_version
  vm_size               = each.value.vm_size

  priority        = each.value.priority
  spot_max_price  = each.value.priority == "Spot" ? -1 : null
  eviction_policy = each.value.eviction_policy
  node_taints     = each.value.node_taints

  node_labels = each.value.node_labels

  tags = each.value.tags
}
