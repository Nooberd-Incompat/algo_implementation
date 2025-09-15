import json
from web3 import Web3
# --- MODIFIED: Import Account and message encoding functions ---
from eth_account import Account
from eth_account.messages import encode_defunct

# --- Configuration ---
GANACHE_URL = "http://127.0.0.1:7545"
ETH_PRICE_DOLLARS = 3000.0

def run_settlement(net_balances_dollars: dict):
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not w3.is_connected():
        raise ConnectionError("Could not connect to Ganache. Is it running?")
    
    accounts = w3.eth.accounts
    print(f"Connected to Ganache. Using {len(accounts)} accounts.")

    # ... (Balance preparation logic is unchanged) ...
    print("\n--- Preparing Balances for Smart Contract ---")
    balances_eth_float = {name: bal / ETH_PRICE_DOLLARS for name, bal in net_balances_dollars.items()}
    balances_wei_unadjusted = [int(bal * 10**18) for bal in balances_eth_float.values()]
    residual_error_wei = sum(balances_wei_unadjusted)
    print(f"Calculated residual error from int() truncation: {residual_error_wei} wei")
    balances_wei_adjusted = balances_wei_unadjusted.copy()
    balances_wei_adjusted[-1] -= residual_error_wei
    if sum(balances_wei_adjusted) != 0:
        raise RuntimeError("CRITICAL: Wei balance correction failed. Sum is not zero.")
    print("Balances in wei have been successfully adjusted to sum to zero.")
    
    with open("compiled_contract.json", "r") as f:
        contract_interface = json.load(f)
    
    bytecode = contract_interface["bytecode"]
    abi = contract_interface["abi"]
    
    print("\nDeploying contract...")
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    participant_names = list(net_balances_dollars.keys())
    participant_addresses = accounts[:len(participant_names)]
    balances_in_wei = balances_wei_adjusted

    tx_hash = Contract.constructor(participant_addresses, balances_in_wei).transact({'from': accounts[0]})
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    print(f"Contract deployed at: {contract_address}")
    
    contract_instance = w3.eth.contract(address=contract_address, abi=abi)
    
    # ... (Funding logic is unchanged) ...
    print("\nPayers are depositing funds...")
    for i, name in enumerate(participant_names):
        if balances_in_wei[i] < 0:
            # ...
            tx_hash = contract_instance.functions.deposit().transact({'from': participant_addresses[i], 'value': abs(balances_in_wei[i])})
            w3.eth.wait_for_transaction_receipt(tx_hash)
    print("All funds deposited into the contract.")

    print("\nSimulating OFF-CHAIN agreement and signing...")
    final_balances_off_chain = [0] * len(participant_names)
    final_nonce = 1 

    # --- !! CRITICAL FIX IS HERE !! ---
    # 1. Get the raw hash of the data from the contract
    data_hash = contract_instance.functions.getDataHash(final_balances_off_chain, final_nonce).call()
    
    # 2. Use the standard library to add the EIP-191 prefix
    signable_message = encode_defunct(primitive=data_hash)
    
    private_keys = [
        "0xbcc8c52b69e86d97eb4a3dcfae554f757621d5295b0c446ed846e3506f8e8688",
        "0x93eb67f9964311163baf33030683b88ac85646f132a4505627bf45d73f9343d7",
        "0xfd2697890be174e29faa7bddf262cdc70a46999c43c6c95c9a6561e4aa16f4e0",
        "0x3c6bee37626fc28a3d92919597cc6c9e8ed8213744461a0518ae9aa0411264dd",
        "0xda97a9c15ece7cca54aa483ae985b6594f6ce0c6d5f0e1b29db6fe3d10ab2180",
        "0xcea5357a6a8e700fc86accc2f9c8e1ac7a2a4827c76efe02f901f57bbca6c469",
        "0x536ed2a1dce032e38adb568d087788bc3f3f5afdbfce2c19d764af5160f2b5b7",
        "0x6579f145dabc0bf0162d48f121b1979f36729634a89db236cd3206508b7db439",
        "0x800ab91f27a589a0f01cb746c7030d1b32e6475a2bb918fb18890ccadd6e242a",
        "0xd0a400eec33132eb2781cb93c9ac1cf819a210c6266bf313e594c2535b1f84ac",
    ]

    if "0x..." in private_keys[0]:
        raise ValueError("Please replace placeholder private keys in scripts/settlement.py")
    
    signatures = []
    # 3. Sign the correctly prefixed message using the standard Account.sign_message
    for key in private_keys:
        signed_message = Account.sign_message(signable_message, private_key=key)
        signatures.append(signed_message.signature)
    print("All participants have signed the final state.")
    
    print("\nClosing the channel with the final signed state...")
    initial_receiver_balances = {addr: w3.eth.get_balance(addr) for i, addr in enumerate(participant_addresses) if balances_in_wei[i] > 0}

    tx_hash = contract_instance.functions.closeChannel(final_balances_off_chain, signatures).transact({'from': accounts[0]})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Channel closed successfully! Funds have been distributed.")

    print("\n--- Verification ---")
    for addr, initial_bal in initial_receiver_balances.items():
        name = participant_names[participant_addresses.index(addr)]
        final_bal = w3.eth.get_balance(addr)
        received = final_bal - initial_bal
        print(f"  - {name} received: {Web3.from_wei(received, 'ether')} ETH")


if __name__ == '__main__':
    # This block is for standalone testing
    m_n_values_dollars = { f'Org_{i}': 1 if i % 2 == 0 else -1 for i in range(10) }
    run_settlement(m_n_values_dollars)