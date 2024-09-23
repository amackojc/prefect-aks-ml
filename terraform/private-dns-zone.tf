resource "random_string" "pdns_prefix" {
  length  = 8
  special = false
  keepers = {
    resource_group = azurerm_resource_group.resource_group.name
  }
}

#resource "azurerm_role_assignment" "managed_identity_role_assign_pdns" {
#  scope                = azurerm_private_dns_zone.private_dns_zone.id
#  role_definition_name = "Private DNS Zone Contributor"
#  principal_id         = azurerm_user_assigned_identity.managed_identity_aks.principal_id
#}
#
#resource "azurerm_private_dns_zone" "private_dns_zone" {
#  name                = "${random_string.pdns_prefix.result}.${var.private_dns_zone_name}"
#  resource_group_name = azurerm_resource_group.resource_group.name
#  tags                = var.tags
#}
#
#resource "azurerm_private_dns_zone_virtual_network_link" "private_dns_zone_virtual_network_link" {
#  name                  = "${var.component_name}-pdns-link"
#  resource_group_name   = azurerm_resource_group.resource_group.name
#  private_dns_zone_name = azurerm_private_dns_zone.private_dns_zone.name
#  virtual_network_id    = data.azurerm_virtual_network.virtual_network.id
#}
