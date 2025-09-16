import json
from web3 import Web3
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

    coordinator_address = accounts[0]
    print(f"Coordinator for this settlement: {coordinator_address}")

    # --- Robust Conversion and Correction Logic ---
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
    
    if not bytecode:
        raise ValueError("Bytecode is empty. Please re-run the compile script.")

    print("\nDeploying contract...")
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    participant_names = list(net_balances_dollars.keys())
    participant_addresses = accounts[:len(participant_names)]
    balances_in_wei = balances_wei_adjusted

    tx_hash = Contract.constructor(participant_addresses, balances_in_wei, coordinator_address).transact({'from': coordinator_address})
    deploy_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = deploy_receipt.contractAddress
    print(f"Contract deployed at: {contract_address}")
    
    contract_instance = w3.eth.contract(address=contract_address, abi=abi)
    
    print("\nCoordinator is performing a batch deposit for all payers...")
    payer_addresses = [participant_addresses[i] for i, bal in enumerate(balances_in_wei) if bal < 0]
    total_wei_to_deposit = sum(abs(bal) for bal in balances_in_wei if bal < 0)
    print(f"Coordinator will deposit a total of {Web3.from_wei(total_wei_to_deposit, 'ether')} ETH on behalf of {len(payer_addresses)} payers.")
    tx_hash = contract_instance.functions.batchDeposit(payer_addresses).transact({'from': coordinator_address, 'value': total_wei_to_deposit})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Batch deposit successful. All funds are now locked in the contract.")

    print("\nSimulating OFF-CHAIN agreement and signing...")
    final_balances_off_chain = [0] * len(participant_names)
    final_nonce = 1 
    data_hash = contract_instance.functions.getDataHash(final_balances_off_chain, final_nonce).call()
    signable_message = encode_defunct(primitive=data_hash)
    
    private_keys = [
        "0xfc246f87e67931592ff03913777835669c812557eb68bde5ad412c0f77bb61b9",
        "0x75bb7b440439d8fcba67b9f8120c487a7614a4bc3682fadb8f7f02a948a8a105",
        "0x4885708207ac639db39e260e41c79f6f2602cb9a967250cbae8a23393a3e3bb5",
        "0x4a8b87856cceb224607933b11405380d1662b6a4f0eaf67ff8e0346bada315ea",
        "0x673f3f23d89c0526f9833cb6f965a4d88668397aec9e9bc60122048da429abf8",
        "0x6e9b2452daa97f6f96fb6111a5b7f39cb236e09fac0e150b7af4cb79896c0e70",
        "0x99a82004917255348f55e8df8e694a451636baf3cc4ba98a36205d1fdd637e03",
        "0xc39589cacbc9ed7b373a9ee62895728a3a23454072788e3dd0cb28b03b2ac8aa",
        "0xda5af7678c9b241625fe051671ae823f0a085f7915c8469c8236ab4e5194a63e",
        "0x091edd657c2660ac85e4d9142e64c0095aec3dd70f7a6a465917bbf992bcc7bf",
    ]
    if not private_keys or "0x..." in private_keys[0]:
        raise ValueError("Please replace placeholder private keys in scripts/settlement.py")
    
    signatures = [Account.sign_message(signable_message, private_key=key).signature for key in private_keys]
    print("All participants have signed the final state.")
    
    print("\nClosing the channel with the final signed state...")
    receiver_info = {
        participant_addresses[i]: balances_in_wei[i]
        for i, bal in enumerate(balances_in_wei) if bal > 0
    }
    initial_receiver_balances = {addr: w3.eth.get_balance(addr) for addr in receiver_info.keys()}

    tx_hash = contract_instance.functions.closeChannel(final_balances_off_chain, signatures).transact({'from': coordinator_address})
    close_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Channel closed successfully! Funds have been distributed.")

    # --- NEW: ROBUST VERIFICATION LOGIC ---
    print("\n--- Verification ---")
    
    # Calculate gas cost for the coordinator
    gas_cost = close_receipt.gasUsed * close_receipt.effectiveGasPrice

    for addr, initial_bal in initial_receiver_balances.items():
        name = participant_names[participant_addresses.index(addr)]
        final_bal = w3.eth.get_balance(addr)
        amount_received = receiver_info[addr]
        
        if addr == coordinator_address:
            # For the coordinator, the net change includes the gas fee
            net_change = final_bal - initial_bal
            expected_change = amount_received - gas_cost
            print(f"  - {name} (Coordinator) received: {Web3.from_wei(amount_received, 'ether')} ETH")
            print(f"    - Gas paid: {Web3.from_wei(gas_cost, 'ether')} ETH")
            print(f"    - Net balance change: {Web3.from_wei(net_change, 'ether')} ETH")
            # Verification check
            if net_change != expected_change:
                print("    - WARNING: Coordinator balance change mismatch!")
        else:
            # For other receivers, the change is just the amount received
            received_wei = final_bal - initial_bal
            print(f"  - {name} received: {Web3.from_wei(received_wei, 'ether')} ETH")
            # Verification check
            if received_wei != amount_received:
                 print("    - WARNING: Receiver balance change mismatch!")