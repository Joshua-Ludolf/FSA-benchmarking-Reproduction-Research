import os
import pandas as pd
import argparse
import math
from openai import OpenAI

# Set your OpenAI API key from the environment variables
os.environ['OPENAI_API_KEY'] = 'key'

def generate_gpt4o_response(cluster, day, aggregated_data, use_statistics):
    user_message = f"Below is the disk performance data of {cluster} for {day}\n"
    grouped_data = aggregated_data.groupby(['host', 'disk_id'])

    for (host, disk_id), group in grouped_data:
        user_message += f"{host}-{disk_id}: "
        if use_statistics:
            user_message += f"mean: {group['mean'].values[0]}, std: {group['std'].values[0]}, min: {group['min'].values[0]}, max: {group['max'].values[0]}\n"
        else:
            user_message += f"latency\n"
            user_message += '\n'.join([f"{math.ceil(row['latency'])}" for _, row in group.iterrows()])
            user_message += "\n"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a disk data analysis expert. Analyze disk performance time series and report only the most critical disks with significantly abnormal performance compared to peers. Respond strictly in the format 'day, cluster, host, disk, is_faulty' for the few highest risk disks, where is_faulty is T. If there are no faulty disks, do not report anything."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content

def aggregate_data_for_day(perseus_dir, cluster, hosts, day):
    aggregated_data = pd.DataFrame()

    for host in hosts:
        host_dir = os.path.join(perseus_dir, cluster, host)
        file_path = os.path.join(host_dir, f"{day}.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['host'] = host
            data = data.dropna(subset=['latency'])  # Drop rows where latency is NaN
            # Sample data for each disk_id
            sampled_data = data.groupby('disk_id').apply(lambda x: x.sample(n=min(20, len(x)))).reset_index(drop=True)
            sampled_data['latency'] = sampled_data['latency'].apply(math.ceil)
            sampled_data['host'] = host
            aggregated_data = pd.concat([aggregated_data, sampled_data])
    
    return aggregated_data

def aggregate_statistics_for_day(perseus_dir, cluster, hosts, day):
    aggregated_data = pd.DataFrame()

    for host in hosts:
        host_dir = os.path.join(perseus_dir, cluster, host)
        file_path = os.path.join(host_dir, f"{day}.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['host'] = host
            data = data.dropna(subset=['latency'])  # Drop rows where latency is NaN
            stats = data.groupby('disk_id')['latency'].agg(['mean', 'std', 'min', 'max']).reset_index()
            stats['host'] = host
            stats['latency'] = stats['mean']
            aggregated_data = pd.concat([aggregated_data, stats])
    
    return aggregated_data

def process_Perseus(perseus_dir, cluster_host_mapping, use_statistics):
    for cluster, hosts in cluster_host_mapping.items():
        host_dir = os.path.join(perseus_dir, cluster, hosts[0])
        files = [f for f in os.listdir(host_dir) if f.endswith('.csv') and len(f) == 14]
        days = sorted({f.rstrip('.csv') for f in files})
        
        for day in days:
            all_predictions = set()
            # Split hosts into batches of 50
            for i in range(0, len(hosts), 50):
                host_batch = hosts[i:i+50]
                if use_statistics:
                    aggregated_data = aggregate_statistics_for_day(perseus_dir, cluster, host_batch, day)
                else:
                    aggregated_data = aggregate_data_for_day(perseus_dir, cluster, host_batch, day)

                prediction = generate_gpt4o_response(cluster, day, aggregated_data, use_statistics)
                # Extract faulty disk information from the prediction
                predictions = prediction.strip().split('\n')
                for line in predictions:
                    parts = line.split(',')
                    if len(parts) == 5 and parts[4].strip() == 'T':
                        all_predictions.add((parts[2].strip(), parts[3].strip()))  # (host, disk)

            # Output the result
            for host in hosts:
                disk_ids = [f"disk{i}" for i in range(1, 13)]
                for disk in disk_ids:
                    is_faulty = 'T' if (host, disk) in all_predictions else 'F'
                    print(f'{day},{cluster},{host},{disk},{is_faulty}')

def main(perseus_dir, input_file, use_statistics):
    df = pd.read_csv(input_file)
    cluster_host_mapping = df.groupby('cluster')['host_name'].apply(list).to_dict()
    process_Perseus(perseus_dir, cluster_host_mapping, use_statistics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Perseus data.')
    parser.add_argument('-p', '--perseus_dir', required=True, help="Path to the directory containing Perseus data files")
    parser.add_argument('-i', '--input_file', required=True, help="Path to the drive file")
    parser.add_argument('-s', '--statistics', action='store_true', help="Use statistics for each host-disk instead of raw data")
    args = parser.parse_args()

    main(args.perseus_dir, args.input_file, args.statistics)
