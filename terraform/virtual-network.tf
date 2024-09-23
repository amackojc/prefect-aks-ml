#data "azurerm_virtual_network" "virtual_network" {
#  name                = var.virtual_network_name
#  resource_group_name = var.virtual_network_resource_group_name
#}
#
#resource "azurerm_role_assignment" "managed_identity_role_assign_vnet" {
#  scope                = azurerm_subnet.subnet.id
#  role_definition_name = "Network Contributor"
#  principal_id         = azurerm_user_assigned_identity.managed_identity_aks.principal_id
#}
#
#resource "azurerm_subnet" "subnet" {
#  name                 = "${var.component_name}-subnet"
#  resource_group_name  = data.azurerm_virtual_network.virtual_network.resource_group_name
#  virtual_network_name = data.azurerm_virtual_network.virtual_network.name
#  address_prefixes     = var.subnet_prefixes
#}
