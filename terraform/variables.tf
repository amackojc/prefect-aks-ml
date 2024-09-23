variable "location" {
  type = string
}

variable "component_name" {
  type = string
}

variable "kubernetes_version" {
  type = string
}

#variable "subnet_prefixes" {
#  type = list(string)
#}

variable "aks_clusters" {
  type = map(
    object({
      identity_type             = string
      default_node_pool_name    = string
      default_node_pool_vm_size = string
      os_disk_size_gb           = string
      os_disk_type              = string
      node_taints               = list(string)
      node_labels               = map(string)
      tags                      = map(string)
      network_plugin            = optional(string, "azure")
      outbound_type             = optional(string, "loadBalancer")
      #outbound_type             = optional(string, "userAssignedNATGateway")

      node_count = number
    })
  )
}

variable "ssh_public_key" {
  type = string
}

variable "admin_username" {
  type = string
}

variable "node_pools" {
  type = map(
    object({
      vm_size             = string
      cluster             = string
      eviction_policy     = string
      max_count           = number
      min_count           = number
      enable_auto_scaling = bool
      node_taints         = list(string)
      node_labels         = map(string)
      tags                = map(string)

      priority       = optional(string, "Spot")
      node_count     = optional(number, 1)
      spot_max_price = optional(number, -1)
    })
  )
}

variable "tags" {
  type = map(string)
}

#variable "private_dns_zone_name" {
#  type = string
#}

#variable "virtual_network_name" {
#  type = string
#}

#variable "virtual_network_resource_group_name" {
#  type = string
#}

variable "backend_storage_account_name" {
  type = string
}
