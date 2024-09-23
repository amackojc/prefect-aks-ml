terraform {
  backend "azurerm" {
    resource_group_name  = "aks-prefect-rg"
    storage_account_name = "aksprefecttfstate"
    container_name       = "tfstate"
    key                  = "prefect.terraform.tfstate"
    #use_azuread_auth     = true
  }
}
