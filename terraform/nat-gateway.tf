#resource "azurerm_public_ip" "nat_gateway_public_ip" {
#  name                = "${var.component_name}-pip"
#  resource_group_name = azurerm_resource_group.resource_group.name
#  location            = azurerm_resource_group.resource_group.location
#  allocation_method   = "Static"
#  sku                 = "Standard"
#  sku_tier            = "Regional"
#
#  tags = var.tags
#}
#
#resource "azurerm_nat_gateway" "nat_gateway" {
#  name                    = "${var.component_name}-ng"
#  resource_group_name     = azurerm_resource_group.resource_group.name
#  location                = azurerm_resource_group.resource_group.location
#  sku_name                = "Standard"
#  idle_timeout_in_minutes = 10
#
#  tags = var.tags
#}
#
#resource "azurerm_nat_gateway_public_ip_association" "nat_gateway_public_ip_association" {
#  nat_gateway_id       = azurerm_nat_gateway.nat_gateway.id
#  public_ip_address_id = azurerm_public_ip.nat_gateway_public_ip.id
#}
#
#resource "azurerm_subnet_nat_gateway_association" "nat_gateway_subnet_association" {
#  subnet_id      = azurerm_subnet.subnet.id
#  nat_gateway_id = azurerm_nat_gateway.nat_gateway.id
#}
#
