terraform {
  required_version = ">=1.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.106.1"
    }
    random = {
      source  = "hashicorp/random"
      version = "~>3.6.2"
    }
  }
}

provider "azurerm" {
  features {}
}
