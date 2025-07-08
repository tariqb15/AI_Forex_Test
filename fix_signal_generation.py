#!/usr/bin/env python3
"""
Fix Signal Generation
Create a simplified signal generation method that works with detected market structures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forex_engine import AgenticForexEngine
from datetime import datetime, timedelta
import MetaTrader5 as mt5

def create_simplified_signal_method():
    """
    Create a simplified signal generation method that actually produces signals
    from the detected market structures
    """
    
    signal_method_code = '''
    def _generate_simplified_trading_signals(self, analysis: Dict) -> List:
        """Generate trading signals with simplified, more permissive criteria"""
        signals = []
        current_time = self._get_broker_time().replace(tzinfo=None)
        current_price = self.data_collector.get_current_price()
        
        # Extract current price value if it's a dict
        if isinstance(current_price, dict):
            current_price = current_price.get('bid', current_price.get('ask', 1.0))
        
        logger.info(f"🎯 Simplified Signal Generation - Current Price: {current_price}")
        
        # Process H1 and H4 timeframes with relaxed criteria
        for tf in ['H1', 'H4']:
            if tf not in analysis.get('timeframe_analysis', {}):
                continue
                
            tf_data = analysis['timeframe_analysis'][tf]
            
            # Get basic structures
            order_blocks = tf_data.get('order_blocks', [])
            liquidity_sweeps = tf_data.get('liquidity_sweeps', [])
            fair_value_gaps = tf_data.get('fair_value_gaps', [])
            
            logger.info(f"📊 {tf}: {len(order_blocks)} OBs, {len(liquidity_sweeps)} sweeps, {len(fair_value_gaps)} FVGs")
            
            # Generate signals from order blocks (simplified criteria)
            for ob in order_blocks[-3:]:  # Only check last 3 order blocks
                try:
                    ob_high = float(ob.get('high', 0))
                    ob_low = float(ob.get('low', 0))
                    ob_type = ob.get('type', '')
                    ob_strength = float(ob.get('strength', 0))
                    
                    # Relaxed criteria: strength > 0.5 instead of 0.7+
                    if ob_strength > 0.5 and ob_high > 0 and ob_low > 0:
                        
                        # Check if price is near the order block
                        distance_to_ob = min(
                            abs(current_price - ob_high) / current_price,
                            abs(current_price - ob_low) / current_price
                        )
                        
                        # If price is within 0.5% of order block, generate signal
                        if distance_to_ob < 0.005:  # 0.5% tolerance
                            
                            direction = 'BUY' if 'Bullish' in ob_type else 'SELL'
                            entry_price = ob_low if direction == 'BUY' else ob_high
                            
                            # Simple risk management
                            if direction == 'BUY':
                                stop_loss = ob_low - (ob_high - ob_low) * 0.2
                                take_profit = ob_high + (ob_high - ob_low) * 1.5
                            else:
                                stop_loss = ob_high + (ob_high - ob_low) * 0.2
                                take_profit = ob_low - (ob_high - ob_low) * 1.5
                            
                            # Calculate risk/reward
                            risk = abs(entry_price - stop_loss)
                            reward = abs(take_profit - entry_price)
                            risk_reward = reward / risk if risk > 0 else 0
                            
                            # Relaxed R:R requirement (0.8 instead of 1.2+)
                            if risk_reward > 0.8:
                                signal = {
                                    'signal_type': 'ORDER_BLOCK_RETEST',
                                    'direction': direction,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit': take_profit,
                                    'confidence': min(ob_strength + 0.2, 0.9),  # Boost confidence
                                    'risk_reward': risk_reward,
                                    'timeframe': tf,
                                    'timestamp': current_time,
                                    'reason': f'{tf} Order Block Retest - Strength: {ob_strength:.2f}'
                                }
                                signals.append(signal)
                                logger.info(f"✅ Generated {direction} signal from {tf} OB - R:R: {risk_reward:.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing order block: {e}")
                    continue
            
            # Generate signals from liquidity sweeps + fair value gaps
            for sweep in liquidity_sweeps[-2:]:  # Check last 2 sweeps
                for fvg in fair_value_gaps[-2:]:  # Check last 2 FVGs
                    try:
                        sweep_price = float(sweep.get('price', 0))
                        fvg_high = float(fvg.get('high', 0))
                        fvg_low = float(fvg.get('low', 0))
                        
                        if sweep_price > 0 and fvg_high > 0 and fvg_low > 0:
                            # Check if sweep and FVG are aligned
                            fvg_mid = (fvg_high + fvg_low) / 2
                            
                            # If sweep is near FVG, generate signal
                            if abs(sweep_price - fvg_mid) / current_price < 0.01:  # 1% tolerance
                                
                                direction = 'BUY' if sweep_price < current_price else 'SELL'
                                entry_price = fvg_mid
                                
                                if direction == 'BUY':
                                    stop_loss = fvg_low - (fvg_high - fvg_low) * 0.5
                                    take_profit = fvg_high + (fvg_high - fvg_low) * 2.0
                                else:
                                    stop_loss = fvg_high + (fvg_high - fvg_low) * 0.5
                                    take_profit = fvg_low - (fvg_high - fvg_low) * 2.0
                                
                                risk = abs(entry_price - stop_loss)
                                reward = abs(take_profit - entry_price)
                                risk_reward = reward / risk if risk > 0 else 0
                                
                                if risk_reward > 0.8:  # Relaxed R:R
                                    signal = {
                                        'signal_type': 'SWEEP_FVG_COMBO',
                                        'direction': direction,
                                        'entry_price': entry_price,
                                        'stop_loss': stop_loss,
                                        'take_profit': take_profit,
                                        'confidence': 0.7,  # Fixed confidence
                                        'risk_reward': risk_reward,
                                        'timeframe': tf,
                                        'timestamp': current_time,
                                        'reason': f'{tf} Liquidity Sweep + FVG Alignment'
                                    }
                                    signals.append(signal)
                                    logger.info(f"✅ Generated {direction} signal from {tf} Sweep+FVG combo")
                    
                    except Exception as e:
                        logger.error(f"Error processing sweep+FVG combo: {e}")
                        continue
        
        logger.info(f"🎯 Simplified signal generation complete: {len(signals)} signals")
        return signals
'''
    
    return signal_method_code

def patch_forex_engine():
    """
    Add the simplified signal generation method to the AgenticForexEngine class
    """
    
    print("🔧 Patching AgenticForexEngine with simplified signal generation...")
    
    # Read the current forex_engine.py file
    with open('forex_engine.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create the simplified method
    simplified_method = create_simplified_signal_method()
    
    # Find where to insert the method (before the last method or class end)
    insert_position = content.rfind('def _create_enhanced_market_narrative')
    if insert_position == -1:
        insert_position = content.rfind('class AgenticForexEngine:') + len('class AgenticForexEngine:')
        # Find the next method to insert before it
        next_method = content.find('def ', insert_position)
        if next_method != -1:
            insert_position = next_method
    
    # Insert the simplified method
    new_content = content[:insert_position] + simplified_method + '\n\n    ' + content[insert_position:]
    
    # Also modify the _generate_enhanced_trading_signals method to use simplified version as fallback
    fallback_code = '''
        # If no signals generated with enhanced method, try simplified approach
        if len(enhanced_signals) == 0:
            logger.info("🔄 No enhanced signals found, trying simplified approach...")
            try:
                simplified_signals = self._generate_simplified_trading_signals(analysis)
                logger.info(f"📊 Simplified approach generated {len(simplified_signals)} signals")
                return simplified_signals[:2]  # Return top 2
            except Exception as e:
                logger.error(f"Simplified signal generation failed: {e}")
                return []
'''
    
    # Find the return statement in _generate_enhanced_trading_signals
    enhanced_return = new_content.find('return enhanced_signals[:2]  # Return top 2 highest quality signals')
    if enhanced_return != -1:
        # Insert fallback before the return
        new_content = new_content[:enhanced_return] + fallback_code + '\n        ' + new_content[enhanced_return:]
    
    # Write the modified content back
    with open('forex_engine.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Successfully patched forex_engine.py with simplified signal generation")
    print("📊 The system will now fallback to simplified signals if enhanced method fails")

if __name__ == "__main__":
    print("🚀 Fixing Signal Generation...")
    patch_forex_engine()
    print("\n✅ Signal generation fix completed!")
    print("\n🔄 Please restart the trading system to use the new signal generation logic.")