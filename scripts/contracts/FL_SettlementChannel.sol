// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title FL_SettlementChannel
 * @author Yojith Kaustabh
 * @notice A multi-party state channel to settle payments for a Cross-Silo
 * Federated Learning incentive mechanism. This contract acts as a trusted
 * escrow and settlement layer.
 */
contract FL_SettlementChannel {
    // --- State Variables ---
    enum ChannelState { Open, Closed }
    ChannelState public currentState;
    
    address public immutable coordinator; // The address of the trusted coordinator

    uint256 public constant DURATION = 24 hours;
    uint256 public immutable closeDeadline;

    address[] public participants;
    mapping(address => int256) public netBalances;
    mapping(address => uint256) public depositedFunds;

    uint256 public totalRequiredFromPayers;
    uint256 public nonce;

    // --- Events ---
    event ChannelOpened(address[] participants, int256[] balances, address coordinator);
    event FundsDeposited(address indexed coordinator, uint256 totalAmount);
    event ChannelClosed(uint256 finalNonce);

    /**
     * @notice Deploys the contract, opening the channel.
     * @param _participants An array of all participant addresses.
     * @param _netBalances An array of the corresponding net balances.
     * @param _coordinator The address of the trusted coordinator for batch deposits.
     */
    constructor(address[] memory _participants, int256[] memory _netBalances, address _coordinator) {
        require(_participants.length == _netBalances.length, "Input arrays must have same length");
        
        coordinator = _coordinator;
        participants = _participants;
        int256 balanceSum = 0;
        uint256 requiredDeposit = 0;

        for (uint i = 0; i < _participants.length; i++) {
            netBalances[_participants[i]] = _netBalances[i];
            balanceSum += _netBalances[i];
            if (_netBalances[i] < 0) {
                requiredDeposit += uint256(-_netBalances[i]);
            }
        }

        require(balanceSum == 0, "Net balances must sum to zero");
        totalRequiredFromPayers = requiredDeposit;
        
        currentState = ChannelState.Open;
        closeDeadline = block.timestamp + DURATION;
        nonce = 0;

        emit ChannelOpened(_participants, _netBalances, _coordinator);
    }
    
    /**
     * @notice Allows the trusted coordinator to deposit funds on behalf of multiple payers.
     * @param _payers The list of addresses the coordinator is depositing for.
     */
    function batchDeposit(address[] memory _payers) external payable {
        require(msg.sender == coordinator, "Only the coordinator can call this function");
        require(currentState == ChannelState.Open, "Channel is not open");
        
        uint256 totalAmountFromPayers = 0;
        for (uint i = 0; i < _payers.length; i++) {
            address payer = _payers[i];
            int256 balance = netBalances[payer];
            
            require(balance < 0, "An address in the list is not a payer");
            require(depositedFunds[payer] == 0, "One of the payers has already deposited");

            uint256 requiredAmount = uint256(-balance);
            depositedFunds[payer] = requiredAmount;
            totalAmountFromPayers += requiredAmount;
        }
        
        require(msg.value == totalAmountFromPayers, "Incorrect total ETH sent for batch deposit");
        emit FundsDeposited(msg.sender, msg.value);
    }

    function closeChannel(int256[] memory finalBalances, bytes[] memory signatures) external {
        require(currentState == ChannelState.Open, "Channel is not open");
        require(block.timestamp <= closeDeadline, "Deadline has passed");
        require(address(this).balance >= totalRequiredFromPayers, "All funds not deposited");
        require(signatures.length == participants.length, "Incorrect number of signatures");

        uint256 finalNonce = nonce + 1;
        
        // 1. Get the hash of the raw data
        bytes32 dataHash = getDataHash(finalBalances, finalNonce);
        // 2. Add the standard EIP-191 prefix to match the Python library's behavior
        bytes32 prefixedMessageHash = keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", dataHash));
        
        for (uint i = 0; i < participants.length; i++) {
            // 3. Recover the signer from the prefixed hash
            address signer = recoverSigner(prefixedMessageHash, signatures[i]);
            require(signer == participants[i], "Invalid signature");
        }

        currentState = ChannelState.Closed;
        nonce = finalNonce;

        for (uint i = 0; i < participants.length; i++) {
            address participant = participants[i];
            int256 balance = netBalances[participant];
            if (balance > 0) {
                payable(participant).transfer(uint256(balance));
            }
        }
        
        emit ChannelClosed(finalNonce);
    }

    /**
     * @notice Hashes the raw channel state data. The EIP-191 prefix is applied by the client.
     */
    function getDataHash(int256[] memory _balances, uint256 _nonce) public view returns (bytes32) {
        return keccak256(abi.encode(address(this), _balances, _nonce));
    }

    /**
     * @dev Recovers the address of a signer from a signature.
     */
    function recoverSigner(bytes32 _messageHash, bytes memory _signature) internal pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);
        return ecrecover(_messageHash, v, r, s);
    }

    function splitSignature(bytes memory sig) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
}