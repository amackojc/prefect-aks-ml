resource "azurerm_resource_group" "resource_group" {
  name     = "${var.component_name}-rg"
  location = var.location

  tags = var.tags
}
