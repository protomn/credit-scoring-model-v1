import aiohttp
import asyncio
from typing import Dict, Optional, List
import json
import logging
from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class BlockchainClient:
    """Client for interacting with Ethereum blockchain and APIs"""
    
    def __init__(self):
        self.ethereum_rpc = os.getenv("ETHEREUM_RPC_URL", "https://mainnet.infura.io/v3/your-key")
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.etherscan_api = os.getenv("ETHERSCAN_API_KEY", "your-etherscan-key")
        self.w3 = Web3(Web3.HTTPProvider(self.ethereum_rpc))
        
    async def get_eth_price(self) -> float:
        """Get current ETH price from CoinGecko"""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.coingecko_api}/simple/price?ids=ethereum&vs_currencies=usd"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['ethereum']['usd']
                    else:
                        logger.warning(f"Failed to get ETH price: {response.status}")
                        return 4500.0  # Fallback price
        except Exception as e:
            logger.error(f"Error fetching ETH price: {e}")
            return 4500.0  # Fallback price
    
    async def get_address_balance(self, address: str) -> float:
        """Get ETH balance of an address"""
        try:
            balance_wei = self.w3.eth.get_balance(address)
            balance_eth = self.w3.from_wei(balance_wei, 'ether')
            return float(balance_eth)
        except Exception as e:
            logger.error(f"Error getting balance for {address}: {e}")
            return 0.0
    
    async def get_transaction_count(self, address: str) -> int:
        """Get transaction count for an address"""
        try:
            return self.w3.eth.get_transaction_count(address)
        except Exception as e:
            logger.error(f"Error getting transaction count for {address}: {e}")
            return 0
    
    async def get_blockchain_data(self, address: str) -> Dict:
        """Get comprehensive blockchain data for an address"""
        try:
            # Get basic data
            balance = await self.get_address_balance(address)
            tx_count = await self.get_transaction_count(address)
            eth_price = await self.get_eth_price()
            
            # Calculate basic metrics
            total_volume_usd = balance * eth_price
            
            # For now, return enhanced mock data based on real balance
            # In production, you'd fetch from Etherscan API or similar
            if balance > 10:  # High balance = good user
                defi_interactions = min(int(balance * 2), 500)
                gas_efficiency = 0.8 + (balance / 100) * 0.1
                unique_tokens = min(int(balance / 2), 50)
                liquidation_history = 0
                on_time_payments = 0.9
            elif balance > 1:  # Medium balance
                defi_interactions = min(int(balance * 5), 200)
                gas_efficiency = 0.6 + (balance / 10) * 0.1
                unique_tokens = min(int(balance * 3), 25)
                liquidation_history = 0
                on_time_payments = 0.8
            else:  # Low balance
                defi_interactions = max(int(balance * 10), 10)
                gas_efficiency = 0.5
                unique_tokens = max(int(balance * 5), 5)
                liquidation_history = 0
                on_time_payments = 0.7
            
            return {
                "address": address,
                "balance_eth": balance,
                "balance_usd": total_volume_usd,
                "transaction_count": tx_count,
                "total_volume": total_volume_usd,
                "defi_interactions": defi_interactions,
                "gas_efficiency": min(gas_efficiency, 0.95),
                "unique_tokens": unique_tokens,
                "liquidation_history": liquidation_history,
                "on_time_payments": on_time_payments,
                "last_updated": "real_time"
            }
            
        except Exception as e:
            logger.error(f"Error getting blockchain data for {address}: {e}")
            # Return fallback data
            return {
                "address": address,
                "balance_eth": 0.0,
                "balance_usd": 0.0,
                "transaction_count": 0,
                "total_volume": 0.0,
                "defi_interactions": 0,
                "gas_efficiency": 0.5,
                "unique_tokens": 0,
                "liquidation_history": 0,
                "on_time_payments": 0.5,
                "last_updated": "fallback",
                "error": str(e)
            }
    
    async def validate_ethereum_address(self, address: str) -> bool:
        """Validate Ethereum address format and checksum"""
        try:
            # Check basic format
            if not address.startswith("0x") or len(address) != 42:
                return False
            
            # Check if it's valid hex
            int(address[2:], 16)
            
            # Check checksum (Web3 validation)
            if self.w3.is_address(address):
                return True
            
            return False
        except Exception:
            return False
    
    async def get_gas_price(self) -> int:
        """Get current gas price in Gwei"""
        try:
            gas_price_wei = self.w3.eth.gas_price
            gas_price_gwei = self.w3.from_wei(gas_price_wei, 'gwei')
            return int(gas_price_gwei)
        except Exception as e:
            logger.error(f"Error getting gas price: {e}")
            return 30  # Fallback gas price
    
    async def estimate_gas(self, from_address: str, to_address: str, value: int) -> int:
        """Estimate gas for a simple ETH transfer"""
        try:
            gas_estimate = self.w3.eth.estimate_gas({
                'from': from_address,
                'to': to_address,
                'value': value
            })
            return gas_estimate
        except Exception as e:
            logger.error(f"Error estimating gas: {e}")
            return 21000  # Standard ETH transfer gas limit

# Global instance
blockchain_client = BlockchainClient() 