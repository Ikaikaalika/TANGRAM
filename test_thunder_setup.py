#!/usr/bin/env python3
"""
Quick Thunder Compute setup test
"""

import subprocess
import json

def test_thunder_cli():
    """Test Thunder Compute CLI functionality."""
    print("Testing Thunder Compute CLI...")
    
    try:
        # Test tnr status
        result = subprocess.run(['tnr', 'status', '--no-wait'], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Thunder CLI is working")
            print("Current instances:")
            print(result.stdout)
            
            # Parse for running instances
            lines = result.stdout.strip().split('\n')
            running_instances = []
            
            for line in lines:
                if '│' in line and 'running' in line.lower():
                    parts = [p.strip() for p in line.split('│') if p.strip()]
                    if parts and parts[0].isdigit():
                        running_instances.append(parts[0])
            
            if running_instances:
                print(f"✅ Found {len(running_instances)} running instance(s): {running_instances}")
                return True, running_instances[0]
            else:
                print("⚠️  No running instances found")
                return True, None
        else:
            print(f"❌ Thunder CLI failed: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print("❌ Thunder CLI command timed out")
        return False, None
    except FileNotFoundError:
        print("❌ Thunder CLI (tnr) not found in PATH")
        return False, None
    except Exception as e:
        print(f"❌ Error testing Thunder CLI: {e}")
        return False, None

def test_instance_creation():
    """Test creating a Thunder Compute instance."""
    print("\nTesting instance creation...")
    
    try:
        # Create instance with minimal config
        result = subprocess.run([
            'tnr', 'create', 
            '--gpu', 't4',  # Use smaller GPU for testing
            '--vcpus', '4',  # Valid vCPU count
            '--mode', 'prototyping'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Instance creation successful")
            print(result.stdout)
            
            # Extract instance ID
            output = result.stdout.strip()
            instance_id = None
            for line in output.split('\n'):
                if 'instance' in line.lower():
                    words = line.split()
                    for word in words:
                        if word.isdigit():
                            instance_id = word
                            break
                    if instance_id:
                        break
            
            if not instance_id:
                instance_id = "0"  # Default assumption
            
            print(f"✅ Created instance: {instance_id}")
            return True, instance_id
        else:
            print(f"❌ Instance creation failed: {result.stderr}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error creating instance: {e}")
        return False, None

def test_basic_connection(instance_id):
    """Test basic connection to instance."""
    print(f"\nTesting connection to instance {instance_id}...")
    
    try:
        # Test basic echo command
        result = subprocess.run([
            'tnr', 'connect', instance_id, 
            '--', 'echo', 'Hello from Thunder Compute!'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Connection successful")
            print(f"Remote output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Connection failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 TANGRAM Thunder Compute Setup Test")
    print("=" * 50)
    
    # Test CLI
    cli_works, existing_instance = test_thunder_cli()
    
    if not cli_works:
        print("\n❌ Thunder CLI test failed. Please ensure tnr is installed and authenticated.")
        return
    
    instance_id = existing_instance
    
    # If no existing instance, create one
    if not instance_id:
        created, instance_id = test_instance_creation()
        if not created:
            print("\n❌ Could not create Thunder Compute instance")
            return
    
    # Test connection
    if instance_id:
        connection_works = test_basic_connection(instance_id)
        
        if connection_works:
            print(f"\n✅ Thunder Compute setup complete!")
            print(f"   Instance ID: {instance_id}")
            print(f"   Ready for TANGRAM heavy processing tasks")
            
            print(f"\n💡 Next steps:")
            print(f"   - TANGRAM is now configured to use Thunder Compute")
            print(f"   - Heavy tasks (SAM, COLMAP, PyBullet) will run on Thunder")
            print(f"   - Use: python main.py --input video.mp4 --mode full")
        else:
            print(f"\n⚠️  Instance created but connection failed")
            print(f"   Instance may still be starting up")
    
    print(f"\n📊 Summary:")
    print(f"   Thunder CLI: {'✅' if cli_works else '❌'}")
    print(f"   Instance: {'✅' if instance_id else '❌'} ({instance_id or 'None'})")
    print(f"   Connection: {'✅' if instance_id and connection_works else '❌'}")

if __name__ == "__main__":
    main()